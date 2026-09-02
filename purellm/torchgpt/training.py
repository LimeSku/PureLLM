from dataclasses import dataclass

import torch
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
)

from purellm.torchgpt.model import TinyGPT


@dataclass(frozen=True)
class SchedulerConfig:
    total_steps: int
    warmup_steps: int
    minimum_lr: float

    def __post_init__(self) -> None:
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if not 0 <= self.warmup_steps < self.total_steps:
            raise ValueError("warmup_steps must be non-negative and below total_steps")
        if self.minimum_lr < 0:
            raise ValueError("minimum_lr must be non-negative")


def build_scheduler(
    optimizer: Optimizer,
    config: SchedulerConfig,
) -> LRScheduler:
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=config.total_steps - config.warmup_steps,
        eta_min=config.minimum_lr,
    )
    if config.warmup_steps == 0:
        return cosine

    warmup = LinearLR(
        optimizer,
        start_factor=1 / config.warmup_steps,
        total_iters=config.warmup_steps,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[config.warmup_steps],
    )


def language_model_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError(
            "logits must have shape (batch_size, sequence_length, vocab_size)"
        )

    if targets.shape != logits.shape[:2]:
        raise ValueError("targets must have shape (batch_size, sequence_length)")

    vocab_size = logits.shape[-1]

    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
    )


@torch.compile
def compiled_training_forward(
    model: TinyGPT,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    return model(token_ids, use_cache=False)


def train_language_model_step(
    model: TinyGPT,
    optimizer: Optimizer,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    max_grad_norm: float | None = 1.0,
    autocast_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if x_batch.shape != y_batch.shape:
        raise ValueError("x_batch and y_batch must have the same shape")

    if max_grad_norm is not None and max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=x_batch.device.type,
        dtype=autocast_dtype,
        enabled=autocast_dtype is not None,
    ):
        logits = compiled_training_forward(model, x_batch)
        loss = language_model_loss(logits, y_batch)

    loss.backward()

    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=max_grad_norm,
        )

    optimizer.step()

    return loss.detach()
