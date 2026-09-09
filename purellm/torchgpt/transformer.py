import torch
from torch import nn
from torch.nn import functional as F

from purellm.torchgpt.attention import MultiHeadCausalSelfAttention
from purellm.torchgpt.position import PositionEncoding


def create_normalization(embedding_dim: int, normalization: str) -> nn.Module:
    if normalization == "layernorm":
        return nn.LayerNorm(embedding_dim)
    if normalization == "rmsnorm":
        return nn.RMSNorm(embedding_dim, eps=1e-5)
    raise ValueError(f"unsupported normalization: {normalization!r}")


class FeedForward(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, init_std: float = 0.02):
        super().__init__()
        self.W1 = nn.Linear(embedding_dim, 2 * hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, embedding_dim, bias=False)

        nn.init.normal_(self.W1.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.W2.weight, mean=0.0, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.W1(x).chunk(2, dim=-1)
        return self.W2(F.silu(gate) * value)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
        init_std: float = 0.02,
        dropout: float = 0.0,
        position_encoding: PositionEncoding = "learned",
        normalization: str = "layernorm",
    ) -> None:
        super().__init__()
        self.ln1 = create_normalization(embedding_dim, normalization)
        self.attention = MultiHeadCausalSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            init_std=init_std,
            dropout=dropout,
            position_encoding=position_encoding,
        )

        self.ln2 = create_normalization(embedding_dim, normalization)
        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim, hidden_dim=hidden_dim, init_std=init_std
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.feed_forward_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        x = x + self.attention_dropout(self.attention(self.ln1(x), use_cache=use_cache))
        x = x + self.feed_forward_dropout(self.feed_forward(self.ln2(x)))
        return x
