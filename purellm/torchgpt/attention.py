import torch
from torch import nn
from torch.nn import functional as F

from purellm.torchgpt.position import PositionEncoding, RotaryEmbedding

# class CausalSelfAttentionHead(nn.Module):
#     def __init__(self, embedding_dim: int, head_dim: int, init_std: float = 0.02):
#         super().__init__()
#         self.embedding_dim = embedding_dim
#         self.head_dim = head_dim

#         self.W_query = nn.Linear(embedding_dim, head_dim, bias=False)
#         self.W_key = nn.Linear(embedding_dim, head_dim, bias=False)
#         self.W_value = nn.Linear(embedding_dim, head_dim, bias=False)
#         nn.init.normal_(self.W_query.weight, mean=0.0, std=init_std)
#         nn.init.normal_(self.W_key.weight, mean=0.0, std=init_std)
#         nn.init.normal_(self.W_value.weight, mean=0.0, std=init_std)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         Q = self.W_query(x)
#         K = self.W_key(x)
#         V = self.W_value(x)

#         attention_scores = Q @ K.transpose(-2, -1)
#         attention_scores = attention_scores / (self.head_dim**0.5)

#         sequence_length = x.shape[-2]
#         mask = torch.triu(
#             torch.ones(sequence_length, sequence_length, dtype=bool, device=x.device),
#             diagonal=1,
#         )
#         # attention_scores = np.where(mask, -np.inf, attention_scores)
#         attention_scores = attention_scores.masked_fill(mask, float("-inf"))
#         # attention_weights = self._softmax(attention_scores)
#         attention_weights = F.softmax(attention_scores, dim=-1)

#         return attention_weights @ V


class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        init_std: float = 0.02,
        dropout: float = 0.0,
        position_encoding: PositionEncoding = "learned",
    ) -> None:
        super().__init__()

        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if position_encoding not in ("learned", "rope"):
            raise ValueError(f"unsupported position encoding: {position_encoding!r}")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.rotary_embedding = (
            RotaryEmbedding(self.head_dim) if position_encoding == "rope" else None
        )
        self.qkv = nn.Linear(
            embedding_dim,
            3 * embedding_dim,
            bias=False,
        )
        self.W_output = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False,
        )
        nn.init.normal_(
            self.qkv.weight,
            mean=0.0,
            std=init_std,
        )

        nn.init.normal_(
            self.W_output.weight,
            mean=0.0,
            std=init_std,
        )
        self.dropout = dropout

        # register KV cache
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)

    def reset_cache(self) -> None:
        self.cache_k = None
        self.cache_v = None

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape
        query, key, value = self.qkv(x).chunk(
            3,
            dim=-1,
        )
        query = query.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        key = key.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        value = value.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        past_length = (
            self.cache_k.shape[-2] if use_cache and self.cache_k is not None else 0
        )

        if past_length and sequence_length != 1:
            raise ValueError("cached decoding expects exactly one new token")

        if self.rotary_embedding is not None:
            query, key = self.rotary_embedding(
                query,
                key,
                position_offset=past_length,
            )

        if use_cache:
            if self.cache_k is not None:
                key = torch.cat((self.cache_k, key), dim=-2)
                value = torch.cat((self.cache_v, value), dim=-2)

            self.cache_k = key.detach()
            self.cache_v = value.detach()

        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            # is_causal=True,
            is_causal=past_length == 0,
            dropout_p=self.dropout if self.training else 0.0,
        )
        output = output.transpose(1, 2).reshape(
            batch_size,
            sequence_length,
            self.embedding_dim,
        )
        return self.W_output(output)
