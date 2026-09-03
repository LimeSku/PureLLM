import argparse
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Any, Literal

import tomllib


@dataclass(frozen=True, slots=True)
class TinyGPTConfig:
    context_length: int
    embedding_dim: int
    num_heads: int
    num_layers: int
    hidden_dim: int
    dropout: float = 0.1
    init_std: float = 0.02
    tie_embeddings: bool = True
    position_encoding: Literal["learned", "rope"] = "rope"

    def __post_init__(self) -> None:
        for name in (
            "context_length",
            "embedding_dim",
            "num_heads",
            "num_layers",
            "hidden_dim",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"model.{name} must be positive")
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError("model.embedding_dim must be divisible by model.num_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError(
                "model.dropout must be between 0 inclusive and 1 exclusive"
            )
        if self.init_std <= 0:
            raise ValueError("model.init_std must be positive")
        if self.position_encoding not in ("learned", "rope"):
            raise ValueError(
                f"unsupported model.position_encoding: {self.position_encoding!r}"
            )


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int
    max_steps: int
    batch_size: int
    learning_rate: float
    minimum_learning_rate: float = 3e-5
    warmup_steps: int = 300
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    precision: Literal["fp32", "bf16"] = "fp32"
    seed: int = 42
    log_every: int = 100
    eval_every: int = 100
    eval_batches: int = 10
    save_every: int = 500

    def __post_init__(self) -> None:
        for name in (
            "epochs",
            "max_steps",
            "batch_size",
            "log_every",
            "eval_every",
            "eval_batches",
            "save_every",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"training.{name} must be positive")
        if self.learning_rate <= 0:
            raise ValueError("training.learning_rate must be positive")
        if not 0 <= self.minimum_learning_rate <= self.learning_rate:
            raise ValueError(
                "training.minimum_learning_rate must be between 0 and learning_rate"
            )
        if not 0 <= self.warmup_steps < self.max_steps:
            raise ValueError("training.warmup_steps must be below training.max_steps")
        if self.weight_decay < 0:
            raise ValueError("training.weight_decay must be non-negative")
        if self.max_grad_norm <= 0:
            raise ValueError("training.max_grad_norm must be positive")
        if self.precision not in ("fp32", "bf16"):
            raise ValueError(f"unsupported training.precision: {self.precision!r}")


@dataclass(frozen=True, slots=True)
class DataConfig:
    train_path: Path
    validation_path: Path | None = None
    validation_fraction: float | None = 0.1
    max_train_characters: int | None = None
    max_validation_characters: int | None = None

    def __post_init__(self) -> None:
        if (self.validation_path is None) == (self.validation_fraction is None):
            raise ValueError(
                "data must define exactly one of validation_path or validation_fraction"
            )
        if (
            self.validation_fraction is not None
            and not 0 < self.validation_fraction < 1
        ):
            raise ValueError("data.validation_fraction must be between 0 and 1")
        for name in (
            "max_train_characters",
            "max_validation_characters",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"data.{name} must be positive when set")


@dataclass(frozen=True, slots=True)
class TokenizerConfig:
    type: str = "character"
    vocab_size: int | None = None
    max_fit_characters: int | None = None

    def __post_init__(self) -> None:
        if not self.type:
            raise ValueError("tokenizer.type must not be empty")
        if self.type == "bpe":
            if self.vocab_size is None:
                raise ValueError("tokenizer.vocab_size is required for BPE")
            if self.vocab_size < 256:
                raise ValueError("tokenizer.vocab_size must be at least 256")
        elif self.vocab_size is not None:
            raise ValueError("tokenizer.vocab_size is only supported for BPE")
        if self.max_fit_characters is not None and self.max_fit_characters <= 0:
            raise ValueError("tokenizer.max_fit_characters must be positive when set")
        if (
            self.type not in ("character", "bpe")
            and self.max_fit_characters is not None
        ):
            raise ValueError(
                "tokenizer.max_fit_characters is only supported for fitted tokenizers"
            )


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    name: str
    output_dir: Path
    model_family: Literal["tinygpt"]
    model: TinyGPTConfig
    tokenizer: TokenizerConfig
    training: TrainingConfig
    data: DataConfig


def load_config(path: Path) -> ExperimentConfig:
    with path.open("rb") as config_file:
        raw = tomllib.load(config_file)

    expected_keys = {
        "name",
        "output_dir",
        "model",
        "tokenizer",
        "training",
        "data",
    }
    unknown_keys = raw.keys() - expected_keys
    if unknown_keys:
        raise ValueError(f"unknown top-level config fields: {sorted(unknown_keys)}")

    try:
        name = raw["name"]
        output_dir = Path(raw["output_dir"])
        model_values = _table(raw, "model")
        tokenizer_values = _table(raw, "tokenizer")
        training_values = _table(raw, "training")
        data_values = _table(raw, "data")
    except (KeyError, TypeError) as error:
        raise ValueError(f"invalid config {path}: {error}") from error

    model_family = model_values.pop("family", None)
    if model_family != "tinygpt":
        raise ValueError(f"unsupported model family: {model_family!r}")

    if "validation_path" in data_values and "validation_fraction" not in data_values:
        data_values["validation_fraction"] = None
    try:
        for field in ("train_path", "validation_path"):
            if field in data_values and data_values[field] is not None:
                data_values[field] = Path(data_values[field])
        return ExperimentConfig(
            name=name,
            output_dir=output_dir,
            model_family=model_family,
            model=TinyGPTConfig(**model_values),
            tokenizer=TokenizerConfig(**tokenizer_values),
            training=TrainingConfig(**training_values),
            data=DataConfig(**data_values),
        )
    except TypeError as error:
        raise ValueError(f"invalid config {path}: {error}") from error


def _table(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config[name]
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a TOML table")
    return value.copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and display a PureLLM config"
    )
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    pprint(load_config(args.config))


if __name__ == "__main__":
    main()
