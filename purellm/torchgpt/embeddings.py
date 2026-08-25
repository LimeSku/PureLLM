import torch
from torch import nn

from purellm.torchgpt.position import PositionEncoding


class TokenPositionEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        ctx_length: int,
        embedding_dim: int,
        dropout: float = 0.0,
        position_encoding: PositionEncoding = "learned",
    ) -> None:
        super().__init__()
        if position_encoding not in ("learned", "rope"):
            raise ValueError(f"unsupported position encoding: {position_encoding!r}")

        self.context_length = ctx_length
        self.position_encoding = position_encoding
        self.dropout = nn.Dropout(dropout)

        self.token_embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        nn.init.normal_(self.token_embedding_layer.weight, mean=0.0, std=0.02)

        if position_encoding == "learned":
            self.pos_embedding_layer: nn.Embedding | None = nn.Embedding(
                ctx_length,
                embedding_dim,
            )
            nn.init.normal_(self.pos_embedding_layer.weight, mean=0.0, std=0.01)
        else:
            self.pos_embedding_layer = None

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch_size, sequence_length)")

        _, sequence_length = token_ids.shape
        if sequence_length > self.context_length:
            raise ValueError(f"sequence length must not exceed {self.context_length}")

        token_embeddings = self.token_embedding_layer(token_ids)
        if self.pos_embedding_layer is None:
            return self.dropout(token_embeddings)

        positions = torch.arange(sequence_length, device=token_ids.device)
        pos_embeddings = self.pos_embedding_layer(positions)
        return self.dropout(token_embeddings + pos_embeddings.unsqueeze(0))
