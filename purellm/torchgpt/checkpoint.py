from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from purellm.tokenization import TextTokenizer, serialize_tokenizer, tokenizer_from_dict
from purellm.torchgpt.model import TinyGPT
from purellm.torchgpt.training import SchedulerConfig, build_scheduler

CHECKPOINT_FORMAT_VERSION = 2


@dataclass(frozen=True)
class LoadedModelCheckpoint:
    model: TinyGPT
    tokenizer: TextTokenizer
    step: int
    best_validation_loss: float


@dataclass(frozen=True)
class LoadedTrainingCheckpoint(LoadedModelCheckpoint):
    optimizer: Optimizer
    scheduler: LRScheduler | None
    scheduler_config: SchedulerConfig | None


def save_training_checkpoint(
    path: Path,
    *,
    model: TinyGPT,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scheduler_config: SchedulerConfig,
    tokenizer: TextTokenizer,
    step: int,
    best_validation_loss: float,
) -> None:
    if step < 0:
        raise ValueError("step must be non-negative")
    if step > scheduler_config.total_steps:
        raise ValueError("step must not exceed the scheduler total_steps")

    device = next(model.parameters()).device

    checkpoint: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_config": {
            "dropout": model.dropout,
            "vocab_size": model.vocab_size,
            "ctx_length": model.ctx_length,
            "embedding_dim": model.embedding_dim,
            "num_heads": model.num_heads,
            "num_layers": model.num_layers,
            "hidden_dim": model.hidden_dim,
            "init_std": model.init_std,
            "tie_embeddings": model.tie_embeddings,
            "position_encoding": model.position_encoding,
        },
        "model_state_dict": model.state_dict(),
        "optimizer_type": type(optimizer).__name__,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_type": type(scheduler).__name__,
        "scheduler_state_dict": scheduler.state_dict(),
        "total_steps": scheduler_config.total_steps,
        "warmup_steps": scheduler_config.warmup_steps,
        "minimum_lr": scheduler_config.minimum_lr,
        **serialize_tokenizer(tokenizer),
        "step": step,
        "best_validation_loss": best_validation_loss,
        "cpu_rng_state": torch.get_rng_state(),
        "device_type": device.type,
        "device_rng_state": _get_device_rng_state(device),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def load_training_checkpoint(
    path: Path,
    *,
    device: torch.device,
) -> LoadedTrainingCheckpoint:
    checkpoint = _load_checkpoint_payload(path, device)
    if checkpoint.get("optimizer_type") != "AdamW":
        raise ValueError(f"unsupported optimizer: {checkpoint.get('optimizer_type')!r}")

    model, tokenizer = _load_model_and_tokenizer(checkpoint, device)

    optimizer = torch.optim.AdamW(model.parameters())
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    scheduler_config = _load_scheduler_config(checkpoint)
    scheduler = _load_scheduler(checkpoint, optimizer, scheduler_config)

    torch.set_rng_state(checkpoint["cpu_rng_state"].cpu())
    _set_device_rng_state(
        device=device,
        saved_device_type=checkpoint["device_type"],
        rng_state=checkpoint["device_rng_state"],
    )

    return LoadedTrainingCheckpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_config=scheduler_config,
        tokenizer=tokenizer,
        step=checkpoint["step"],
        best_validation_loss=checkpoint["best_validation_loss"],
    )


def _load_scheduler(
    checkpoint: dict[str, Any],
    optimizer: Optimizer,
    config: SchedulerConfig | None,
) -> LRScheduler | None:
    scheduler_type = checkpoint.get("scheduler_type")
    scheduler_state = checkpoint.get("scheduler_state_dict")

    if scheduler_type is None and scheduler_state is None:
        return None
    if not isinstance(scheduler_state, dict):
        raise ValueError("invalid scheduler state in checkpoint")
    if config is None:
        raise ValueError("scheduler checkpoint has no scheduler configuration")

    optimizer_learning_rates = [group["lr"] for group in optimizer.param_groups]
    scheduler = build_scheduler(optimizer, config)
    if type(scheduler).__name__ != scheduler_type:
        raise ValueError(f"unsupported scheduler: {scheduler_type!r}")

    scheduler.load_state_dict(scheduler_state)
    for parameter_group, learning_rate in zip(
        optimizer.param_groups,
        optimizer_learning_rates,
        strict=True,
    ):
        parameter_group["lr"] = learning_rate
    return scheduler


def _load_scheduler_config(
    checkpoint: dict[str, Any],
) -> SchedulerConfig | None:
    serialized_config = {
        "total_steps": checkpoint.get("total_steps"),
        "warmup_steps": checkpoint.get("warmup_steps"),
        "minimum_lr": checkpoint.get("minimum_lr"),
    }
    if all(value is None for value in serialized_config.values()):
        return None
    if any(value is None for value in serialized_config.values()):
        raise ValueError("incomplete scheduler configuration in checkpoint")

    return SchedulerConfig(**serialized_config)


def load_model_checkpoint(
    path: Path,
    *,
    device: torch.device,
) -> LoadedModelCheckpoint:
    checkpoint = _load_checkpoint_payload(path, device)
    model, tokenizer = _load_model_and_tokenizer(checkpoint, device)

    return LoadedModelCheckpoint(
        model=model,
        tokenizer=tokenizer,
        step=checkpoint["step"],
        best_validation_loss=checkpoint["best_validation_loss"],
    )


def _load_checkpoint_payload(
    path: Path,
    device: torch.device,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError(f"invalid checkpoint payload: {path}")
    if checkpoint.get("format_version") not in (1, CHECKPOINT_FORMAT_VERSION):
        raise ValueError(
            "unsupported checkpoint format version: "
            f"{checkpoint.get('format_version')!r}"
        )

    return checkpoint


def _load_model_and_tokenizer(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> tuple[TinyGPT, TextTokenizer]:
    model = TinyGPT(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    tokenizer = _load_tokenizer(checkpoint)
    if tokenizer.vocab_size != model.vocab_size:
        raise ValueError(
            "checkpoint tokenizer vocabulary size does not match the model"
        )

    return model, tokenizer


def _load_tokenizer(checkpoint: dict[str, Any]) -> TextTokenizer:
    tokenizer_config = checkpoint.get("tokenizer_config")
    if isinstance(tokenizer_config, dict):
        return tokenizer_from_dict(tokenizer_config)

    tokenizer_type = checkpoint["tokenizer_type"]
    if tokenizer_type == "character":
        characters = checkpoint.get("tokenizer_characters")
        if not isinstance(characters, str):
            raise ValueError("invalid character tokenizer in checkpoint")
        return tokenizer_from_dict({
            "type": "character",
            "version": 1,
            "characters": characters,
        })
    if tokenizer_type == "tiktoken_byte_pair" and isinstance(tokenizer_config, str):
        return tokenizer_from_dict({
            "type": "tiktoken",
            "version": 1,
            "encoding": tokenizer_config,
        })
    raise ValueError("invalid tokenizer config in checkpoint")


def _get_device_rng_state(device: torch.device) -> torch.Tensor | None:
    if device.type == "cuda":
        return torch.cuda.get_rng_state(device)
    if device.type == "mps":
        return torch.mps.get_rng_state()
    return None


def _set_device_rng_state(
    *,
    device: torch.device,
    saved_device_type: str,
    rng_state: torch.Tensor | None,
) -> None:
    if rng_state is None or saved_device_type != device.type:
        return
    if device.type == "cuda":
        torch.cuda.set_rng_state(rng_state.cpu(), device)
    elif device.type == "mps":
        torch.mps.set_rng_state(rng_state.cpu())
