from typing import Literal

import torch
from torch import nn

PositionEncoding = Literal["learned", "rope"]


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError("head_dim must be a positive even number")
        if base <= 0:
            raise ValueError("base must be positive")

        self.head_dim = head_dim
        self.base = base
        exponents = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        inverse_frequencies = base ** (-exponents)
        self.register_buffer(
            "inverse_frequencies",
            inverse_frequencies,
            persistent=False,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query.shape != key.shape:
            raise ValueError("query and key must have the same shape")
        if query.shape[-1] != self.head_dim:
            raise ValueError(f"query and key last dimension must be {self.head_dim}")

        sequence_length = query.shape[-2]
        positions = torch.arange(
            sequence_length,
            dtype=self.inverse_frequencies.dtype,
            device=query.device,
        )
        angles = torch.outer(positions, self.inverse_frequencies)
        broadcast_shape = (1,) * (query.ndim - 2) + angles.shape
        cosine = angles.cos().to(dtype=query.dtype).view(broadcast_shape)
        sine = angles.sin().to(dtype=query.dtype).view(broadcast_shape)

        return (
            self._rotate(query, cosine, sine),
            self._rotate(key, cosine, sine),
        )

    @staticmethod
    def _rotate(
        x: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
    ) -> torch.Tensor:
        even = x[..., 0::2]
        odd = x[..., 1::2]
        rotated_even = even * cosine - odd * sine
        rotated_odd = even * sine + odd * cosine
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)
