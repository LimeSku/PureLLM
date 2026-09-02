import argparse
from pathlib import Path
from time import perf_counter

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader

from purellm.config import (
    DataConfig,
    ExperimentConfig,
    TinyGPTConfig,
    load_config,
)
from purellm.dataset import create_dataloader
from purellm.tokenization import (
    TextTokenizer,
    resolve_tokenizer,
    validate_tokenizer,
)
from purellm.torchgpt.checkpoint import (
    SchedulerConfig,
    create_warmup_cosine_scheduler,
    load_training_checkpoint,
    save_training_checkpoint,
)
from purellm.torchgpt.generation import generate
from purellm.torchgpt.model import TinyGPT
from purellm.torchgpt.training import (
    language_model_loss,
    train_language_model_step,
)
from purellm.utils import create_run_directory, select_device, synchronize_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def read_text(path: Path, max_characters: int | None = None) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"dataset file not found: {path}")

    with path.open(encoding="utf-8") as file:
        return file.read(max_characters)


def load_texts(config: DataConfig) -> tuple[str, str]:
    if config.validation_path is not None:
        training_text = read_text(config.train_path, config.max_train_characters)
        validation_text = read_text(
            config.validation_path,
            config.max_validation_characters,
        )
    else:
        text = read_text(config.train_path)
        validation_fraction = config.validation_fraction
        if validation_fraction is None:
            raise ValueError(
                "data.validation_fraction is required without validation_path"
            )

        split_index = int(len(text) * (1 - validation_fraction))
        training_text = text[:split_index]
        validation_text = text[split_index:]
        if config.max_train_characters is not None:
            training_text = training_text[: config.max_train_characters]
        if config.max_validation_characters is not None:
            validation_text = validation_text[: config.max_validation_characters]

    if not training_text or not validation_text:
        raise ValueError("training and validation data must not be empty")
    return training_text, validation_text


def validate_model(model: TinyGPT, config: TinyGPTConfig) -> None:
    values = {
        "context_length": (model.ctx_length, config.context_length),
        "embedding_dim": (model.embedding_dim, config.embedding_dim),
        "num_heads": (model.num_heads, config.num_heads),
        "num_layers": (model.num_layers, config.num_layers),
        "hidden_dim": (model.hidden_dim, config.hidden_dim),
        "dropout": (model.dropout, config.dropout),
        "init_std": (model.init_std, config.init_std),
        "tie_embeddings": (model.tie_embeddings, config.tie_embeddings),
        "position_encoding": (model.position_encoding, config.position_encoding),
    }
    mismatches = [
        name
        for name, (checkpoint_value, config_value) in values.items()
        if checkpoint_value != config_value
    ]
    if mismatches:
        raise ValueError(
            "checkpoint model does not match the config fields: "
            + ", ".join(mismatches)
        )


def build_scheduler(
    optimizer: Optimizer,
    config: SchedulerConfig,
) -> LRScheduler:
    if config.warmup_steps == 0:
        return CosineAnnealingLR(
            optimizer=optimizer,
            T_max=config.total_steps,
            eta_min=config.minimum_lr,
        )
    return create_warmup_cosine_scheduler(optimizer, config)


@torch.no_grad()
def evaluate_language_model(
    model: TinyGPT,
    data_loader: DataLoader,
    device: torch.device,
    num_batches: int,
    autocast_dtype: torch.dtype | None = None,
) -> float:
    if len(data_loader) == 0:
        raise ValueError("validation DataLoader must contain at least one batch")

    model.eval()
    losses = []
    for batch_index, (x_batch, y_batch) in enumerate(data_loader):
        if batch_index == num_batches:
            break
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=autocast_dtype is not None,
        ):
            logits = model(x_batch)
            losses.append(language_model_loss(logits, y_batch))

    return torch.stack(losses).mean().item()


def train_model(
    *,
    model: TinyGPT,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scheduler_config: SchedulerConfig,
    tokenizer: TextTokenizer,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    sample_prompt_ids: list[int],
    config: ExperimentConfig,
    checkpoint_dir: Path,
    start_step: int,
    best_validation_loss: float,
    autocast_dtype: torch.dtype | None,
) -> None:
    device = next(model.parameters()).device
    training_config = config.training
    steps_per_epoch = len(training_loader)
    sample_max_new_tokens = min(
        100,
        max(1, model.ctx_length - len(sample_prompt_ids)),
    )

    synchronize_device(device)
    interval_started_at = perf_counter()
    last_reported_step = start_step - 1
    step = start_step
    first_epoch = (start_step - 1) // steps_per_epoch + 1

    for epoch in range(first_epoch, training_config.epochs + 1):
        for x_batch, y_batch in training_loader:
            if step > scheduler_config.total_steps:
                return

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            loss = train_language_model_step(
                model=model,
                optimizer=optimizer,
                x_batch=x_batch,
                y_batch=y_batch,
                max_grad_norm=training_config.max_grad_norm,
                autocast_dtype=autocast_dtype,
            )
            scheduler.step()

            should_evaluate = (
                step == 1
                or step % training_config.eval_every == 0
                or step == scheduler_config.total_steps
            )
            should_report = (
                step == 1 or step % training_config.log_every == 0 or should_evaluate
            )
            if should_report:
                synchronize_device(device)
                training_elapsed = perf_counter() - interval_started_at
                interval_steps = step - last_reported_step

            validation_loss = None
            saved_best_checkpoint_path = None
            if should_evaluate:
                validation_loss = evaluate_language_model(
                    model=model,
                    data_loader=validation_loader,
                    device=device,
                    num_batches=training_config.eval_batches,
                    autocast_dtype=autocast_dtype,
                )
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_checkpoint_path = checkpoint_dir / "best.pt"
                    save_training_checkpoint(
                        best_checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scheduler_config=scheduler_config,
                        tokenizer=tokenizer,
                        step=step,
                        best_validation_loss=best_validation_loss,
                    )
                    saved_best_checkpoint_path = best_checkpoint_path

            if should_report:
                tokens_per_second = (
                    interval_steps
                    * training_config.batch_size
                    * model.ctx_length
                    / training_elapsed
                )
                message = (
                    f"[train] epoch {epoch:,}/{training_config.epochs:,} | "
                    f"step {step:>6,}/{scheduler_config.total_steps:,} | "
                    f"loss {loss.item():.4f}"
                )
                if validation_loss is not None:
                    message += f" | val {validation_loss:.4f}"
                message += (
                    f" | lr {scheduler.get_last_lr()[0]:.2e} | "
                    f"{tokens_per_second:,.0f} tok/s | "
                    f"{training_elapsed / interval_steps:.3f}s/step"
                )
                print(message, flush=True)

            if saved_best_checkpoint_path is not None:
                print(
                    f"[checkpoint] best | step {step:,} | "
                    f"validation loss {validation_loss:.4f} | "
                    f"{saved_best_checkpoint_path}"
                )

            if (
                step % training_config.save_every == 0
                or step == scheduler_config.total_steps
            ):
                last_checkpoint_path = checkpoint_dir / "last.pt"
                save_training_checkpoint(
                    last_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scheduler_config=scheduler_config,
                    tokenizer=tokenizer,
                    step=step,
                    best_validation_loss=best_validation_loss,
                )
                print(f"[checkpoint] latest | step {step:,} | {last_checkpoint_path}")

            if should_evaluate:
                with torch.random.fork_rng(
                    devices=[] if device.type == "cpu" else None,
                    device_type=device.type,
                ):
                    torch.manual_seed(training_config.seed)
                    generated_ids = generate(
                        model=model,
                        prompt_ids=sample_prompt_ids,
                        max_new_tokens=sample_max_new_tokens,
                        temperature=0.8,
                        device=device,
                        autocast_dtype=autocast_dtype,
                    )
                print(
                    f"[sample] step {step:,}\n"
                    f"{tokenizer.decode(generated_ids, errors='replace')}\n",
                    flush=True,
                )

            if should_report:
                interval_started_at = perf_counter()
                last_reported_step = step
            step += 1


def print_training_intro(
    *,
    args: argparse.Namespace,
    config: ExperimentConfig,
    run_directory: Path,
    checkpoint_dir: Path,
    device: torch.device,
    model: TinyGPT,
    optimizer: Optimizer,
    scheduler_config: SchedulerConfig,
    tokenizer: TextTokenizer,
    tokenizer_preparation_elapsed: float | None,
    training_character_count: int,
    validation_character_count: int,
    training_sequence_count: int,
    validation_sequence_count: int,
    steps_per_epoch: int,
    start_step: int,
) -> None:
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    tokenizer_description = f"{tokenizer.name} | vocab {tokenizer.vocab_size:,}"
    if tokenizer_preparation_elapsed is not None:
        tokenizer_description += f" | prepared in {tokenizer_preparation_elapsed:.2f}s"

    if scheduler_config.warmup_steps:
        scheduler_description = (
            f"linear warmup {scheduler_config.warmup_steps} steps | "
            f"cosine to {scheduler_config.minimum_lr:.2e}"
        )
    else:
        scheduler_description = f"cosine to {scheduler_config.minimum_lr:.2e}"

    print(
        f"\nPureLLM | {config.name}\n"
        f"  config        {args.config}\n"
        f"  output        {run_directory}\n"
        f"  runtime       {device} | {config.training.precision}\n"
        f"\nData\n"
        f"  source        {config.data.train_path}\n"
        f"  characters    train {training_character_count:,} | "
        f"validation {validation_character_count:,}\n"
        f"  sequences     train {training_sequence_count:,} | "
        f"validation {validation_sequence_count:,}\n"
        f"  tokenizer     {tokenizer_description}\n"
        f"\nModel\n"
        f"  architecture  {config.model_family} | {model.position_encoding} | "
        f"tied embeddings {str(model.tie_embeddings).lower()}\n"
        f"  dimensions    layers {model.num_layers} | dim {model.embedding_dim} | "
        f"heads {model.num_heads} | ff {model.hidden_dim} | "
        f"context {model.ctx_length}\n"
        f"  regularization dropout {model.dropout}\n"
        f"  parameters    {parameter_count:,}\n"
        f"\nTraining\n"
        f"  optimizer     AdamW | lr {optimizer.param_groups[0]['lr']:.2e} | "
        f"weight decay {config.training.weight_decay}\n"
        f"  schedule      {scheduler_description}\n"
        f"  duration      epochs {config.training.epochs:,} | "
        f"max steps {config.training.max_steps:,} | "
        f"planned steps {scheduler_config.total_steps:,}\n"
        f"  batches       {steps_per_epoch:,}/epoch | "
        f"size {config.training.batch_size}\n"
        f"  checkpoints   {checkpoint_dir}"
    )
    if args.resume is not None:
        print(f"  resume        {args.resume} | step {start_step - 1}")
    print()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_config = config.model
    tokenizer_config = config.tokenizer
    training_config = config.training
    data_config = config.data

    torch.manual_seed(training_config.seed)
    device = select_device()
    if training_config.precision == "bf16":
        if device.type not in {"cuda", "mps"}:
            raise ValueError("bf16 training is currently only supported on CUDA or MPS")
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("this CUDA GPU does not support bfloat16")
    autocast_dtype = torch.bfloat16 if training_config.precision == "bf16" else None

    training_text, validation_text = load_texts(data_config)
    if args.resume is None:
        tokenizer_started_at = perf_counter()
        tokenizer = resolve_tokenizer(
            training_text,
            tokenizer_config,
            cache_directory=config.output_dir.parent / "tokenizers",
        )
        tokenizer_preparation_elapsed = perf_counter() - tokenizer_started_at
        model = TinyGPT(
            vocab_size=tokenizer.vocab_size,
            ctx_length=model_config.context_length,
            embedding_dim=model_config.embedding_dim,
            num_heads=model_config.num_heads,
            num_layers=model_config.num_layers,
            hidden_dim=model_config.hidden_dim,
            init_std=model_config.init_std,
            dropout=model_config.dropout,
            tie_embeddings=model_config.tie_embeddings,
            position_encoding=model_config.position_encoding,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            betas=(0.9, 0.95),
        )
        start_step = 1
        best_validation_loss = float("inf")
    else:
        tokenizer_preparation_elapsed = None
        loaded_checkpoint = load_training_checkpoint(args.resume, device=device)
        model = loaded_checkpoint.model
        optimizer = loaded_checkpoint.optimizer
        tokenizer = loaded_checkpoint.tokenizer
        validate_model(model, model_config)
        validate_tokenizer(tokenizer, tokenizer_config)
        if (
            loaded_checkpoint.scheduler is None
            or loaded_checkpoint.scheduler_config is None
        ):
            raise ValueError("checkpoint does not contain a resumable scheduler")
        scheduler = loaded_checkpoint.scheduler
        start_step = loaded_checkpoint.step + 1
        best_validation_loss = loaded_checkpoint.best_validation_loss

    training_loader = create_dataloader(
        training_text,
        tokenizer,
        batch_size=training_config.batch_size,
        max_length=model.ctx_length,
        stride=model.ctx_length,
        shuffle=True,
        drop_last=True,
    )
    validation_loader = create_dataloader(
        validation_text,
        tokenizer,
        batch_size=training_config.batch_size,
        max_length=model.ctx_length,
        stride=model.ctx_length,
        shuffle=False,
        drop_last=False,
    )
    if len(training_loader) == 0:
        raise ValueError("training DataLoader must contain at least one batch")
    if len(validation_loader) == 0:
        raise ValueError("validation DataLoader must contain at least one batch")

    steps_per_epoch = len(training_loader)
    total_steps = min(
        training_config.max_steps,
        training_config.epochs * steps_per_epoch,
    )
    scheduler_config = SchedulerConfig(
        total_steps=total_steps,
        warmup_steps=training_config.warmup_steps,
        minimum_lr=training_config.minimum_learning_rate,
    )
    if args.resume is None:
        scheduler = build_scheduler(optimizer, scheduler_config)
    elif loaded_checkpoint.scheduler_config != scheduler_config:
        raise ValueError("checkpoint scheduler does not match the training config")

    sample_prompt_ids = tokenizer.encode(validation_text)[: min(16, model.ctx_length)]
    training_character_count = len(training_text)
    validation_character_count = len(validation_text)
    del training_text, validation_text

    if args.resume is None:
        run_directory = create_run_directory(config.output_dir)
        (run_directory / "config.toml").write_bytes(args.config.read_bytes())
        checkpoint_dir = run_directory / "checkpoints"
    else:
        checkpoint_dir = args.resume.parent
        run_directory = checkpoint_dir.parent

    print_training_intro(
        args=args,
        config=config,
        run_directory=run_directory,
        checkpoint_dir=checkpoint_dir,
        device=device,
        model=model,
        optimizer=optimizer,
        scheduler_config=scheduler_config,
        tokenizer=tokenizer,
        tokenizer_preparation_elapsed=tokenizer_preparation_elapsed,
        training_character_count=training_character_count,
        validation_character_count=validation_character_count,
        training_sequence_count=len(training_loader.dataset),
        validation_sequence_count=len(validation_loader.dataset),
        steps_per_epoch=steps_per_epoch,
        start_step=start_step,
    )
    train_model(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_config=scheduler_config,
        tokenizer=tokenizer,
        training_loader=training_loader,
        validation_loader=validation_loader,
        sample_prompt_ids=sample_prompt_ids,
        config=config,
        checkpoint_dir=checkpoint_dir,
        start_step=start_step,
        best_validation_loss=best_validation_loss,
        autocast_dtype=autocast_dtype,
    )


if __name__ == "__main__":
    main()
