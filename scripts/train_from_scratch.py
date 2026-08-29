import argparse
from math import ceil
from pathlib import Path
from time import perf_counter

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler

from purellm.config import DataConfig, ExperimentConfig, TinyGPTConfig, load_config
from purellm.tokenization import (
    BytePairTokenizer,
    CharacterTokenizer,
    TextTokenizer,
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


def encode_text(
    tokenizer: TextTokenizer,
    text: str,
    device: torch.device,
    split: str,
) -> torch.Tensor:
    try:
        encoded_text = tokenizer.encode(text)
    except KeyError as error:
        missing_character = error.args[0]
        raise ValueError(
            f"{split} text contains a character absent from the tokenizer "
            f"vocabulary: {missing_character!r}"
        ) from error

    return torch.tensor(encoded_text, dtype=torch.long, device=device)


def fit_tokenizer(text: str, config: DataConfig) -> TextTokenizer:
    if config.tokenizer == "character":
        return CharacterTokenizer().fit(text)

    training_text = (
        text
        if config.tokenizer_training_characters is None
        else text[: config.tokenizer_training_characters]
    )
    return BytePairTokenizer().fit(
        training_text,
        vocab_size=config.tokenizer_vocab_size,
    )


def tokenizer_name(tokenizer: TextTokenizer) -> str:
    if isinstance(tokenizer, CharacterTokenizer):
        return "character"
    if isinstance(tokenizer, BytePairTokenizer):
        return "bpe"
    raise TypeError(f"unsupported tokenizer: {type(tokenizer).__name__}")


def validate_tokenizer(tokenizer: TextTokenizer, config: DataConfig) -> None:
    if tokenizer_name(tokenizer) != config.tokenizer:
        raise ValueError("checkpoint tokenizer does not match the data config")


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


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


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


def create_run_directory(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_number = (
        max(
            (
                int(path.name)
                for path in output_dir.iterdir()
                if path.is_dir() and path.name.isdigit()
            ),
            default=0,
        )
        + 1
    )

    while True:
        run_directory = output_dir / str(run_number)
        try:
            run_directory.mkdir()
        except FileExistsError:
            run_number += 1
        else:
            return run_directory


def sample_batch(
    token_ids: torch.Tensor,
    batch_size: int,
    context_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(token_ids) - context_length
    if max_start <= 0:
        raise ValueError("token_ids must contain more tokens than context_length")

    starts = torch.randint(
        0,
        max_start,
        (batch_size,),
        device=token_ids.device,
    )
    offsets = torch.arange(context_length, device=token_ids.device)
    indices = starts[:, None] + offsets[None, :]
    return token_ids[indices], token_ids[indices + 1]


@torch.no_grad()
def evaluate_language_model(
    model: TinyGPT,
    token_ids: torch.Tensor,
    batch_size: int,
    context_length: int,
    num_batches: int,
    autocast_dtype: torch.dtype | None = None,
) -> float:
    model.eval()
    losses = []

    for _ in range(num_batches):
        x_batch, y_batch = sample_batch(
            token_ids=token_ids,
            batch_size=batch_size,
            context_length=context_length,
        )
        with torch.autocast(
            device_type=token_ids.device.type,
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
    train_token_ids: torch.Tensor,
    validation_token_ids: torch.Tensor,
    config: ExperimentConfig,
    checkpoint_dir: Path,
    steps_per_epoch: int,
    start_step: int,
    best_validation_loss: float,
    autocast_dtype: torch.dtype | None,
) -> None:
    device = train_token_ids.device
    training_config = config.training
    sample_prompt_ids = validation_token_ids[: min(16, model.ctx_length)].tolist()
    sample_max_new_tokens = min(
        100,
        max(1, model.ctx_length - len(sample_prompt_ids)),
    )

    synchronize_device(device)
    interval_started_at = perf_counter()
    last_reported_step = start_step - 1

    for step in range(start_step, scheduler_config.total_steps + 1):
        epoch = (step - 1) // steps_per_epoch + 1
        x_batch, y_batch = sample_batch(
            token_ids=train_token_ids,
            batch_size=training_config.batch_size,
            context_length=model.ctx_length,
        )
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
                token_ids=validation_token_ids,
                batch_size=training_config.batch_size,
                context_length=model.ctx_length,
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
    tokenizer_training_elapsed: float | None,
    training_character_count: int,
    validation_character_count: int,
    training_token_count: int,
    validation_token_count: int,
    steps_per_epoch: int,
    start_step: int,
) -> None:
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    tokenizer_description = (
        f"{tokenizer_name(tokenizer)} | vocab {tokenizer.vocab_size:,}"
    )
    if tokenizer_training_elapsed is not None:
        tokenizer_description += f" | fitted in {tokenizer_training_elapsed:.2f}s"

    if scheduler_config.warmup_steps:
        scheduler_description = (
            f"linear warmup {scheduler_config.warmup_steps} steps | "
            f"cosine to {scheduler_config.minimum_lr:.2e}"
        )
    else:
        scheduler_description = f"cosine to {scheduler_config.minimum_lr:.2e}"

    print(
        f"\nPureLLM from scratch | {config.name}\n"
        f"  config        {args.config}\n"
        f"  output        {run_directory}\n"
        f"  runtime       {device} | {config.training.precision}\n"
        f"\nData\n"
        f"  source        {config.data.train_path}\n"
        f"  characters    train {training_character_count:,} | "
        f"validation {validation_character_count:,}\n"
        f"  tokens        train {training_token_count:,} | "
        f"validation {validation_token_count:,}\n"
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
        f"  duration      virtual epochs {config.training.epochs:,} | "
        f"max steps {config.training.max_steps:,} | "
        f"planned steps {scheduler_config.total_steps:,}\n"
        f"  batches       {steps_per_epoch:,}/virtual epoch | "
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
    training_config = config.training
    data_config = config.data

    torch.manual_seed(training_config.seed)
    device = select_device()
    if training_config.precision == "bf16":
        if device.type not in {"cuda", "mps"}:
            raise ValueError("bf16 training is currently only supported on CUDA or MPS")
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("this CUDA GPU does not support bfloat16")
    autocast_dtype = (
        torch.bfloat16 if training_config.precision == "bf16" else None
    )

    training_text, validation_text = load_texts(data_config)
    if args.resume is None:
        tokenizer_started_at = perf_counter()
        tokenizer = fit_tokenizer(training_text, data_config)
        tokenizer_training_elapsed = perf_counter() - tokenizer_started_at
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
        tokenizer_training_elapsed = None
        loaded_checkpoint = load_training_checkpoint(args.resume, device=device)
        model = loaded_checkpoint.model
        optimizer = loaded_checkpoint.optimizer
        tokenizer = loaded_checkpoint.tokenizer
        validate_model(model, model_config)
        validate_tokenizer(tokenizer, data_config)
        if (
            loaded_checkpoint.scheduler is None
            or loaded_checkpoint.scheduler_config is None
        ):
            raise ValueError("checkpoint does not contain a resumable scheduler")
        scheduler = loaded_checkpoint.scheduler
        start_step = loaded_checkpoint.step + 1
        best_validation_loss = loaded_checkpoint.best_validation_loss

    train_token_ids = encode_text(tokenizer, training_text, device, "training")
    validation_token_ids = encode_text(
        tokenizer,
        validation_text,
        device,
        "validation",
    )
    tokens_per_step = training_config.batch_size * model.ctx_length
    steps_per_epoch = ceil((len(train_token_ids) - 1) / tokens_per_step)
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
        tokenizer_training_elapsed=tokenizer_training_elapsed,
        training_character_count=training_character_count,
        validation_character_count=validation_character_count,
        training_token_count=len(train_token_ids),
        validation_token_count=len(validation_token_ids),
        steps_per_epoch=steps_per_epoch,
        start_step=start_step,
    )
    train_model(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_config=scheduler_config,
        tokenizer=tokenizer,
        train_token_ids=train_token_ids,
        validation_token_ids=validation_token_ids,
        config=config,
        checkpoint_dir=checkpoint_dir,
        steps_per_epoch=steps_per_epoch,
        start_step=start_step,
        best_validation_loss=best_validation_loss,
        autocast_dtype=autocast_dtype,
    )


if __name__ == "__main__":
    main()
