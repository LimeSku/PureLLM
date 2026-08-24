import torch
from torch import nn

from purellm.llm.torchgpt.embeddings import TokenPositionEmbedding
from purellm.llm.torchgpt.position import PositionEncoding
from purellm.llm.torchgpt.transformer import TransformerBlock


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        ctx_length: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        hidden_dim: int,
        init_std: float = 0.02,
        dropout: float = 0.0,
        tie_embeddings: bool = False,
        position_encoding: PositionEncoding = "learned",
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.ctx_length = ctx_length
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.init_std = init_std
        self.dropout = dropout
        self.tie_embeddings = tie_embeddings
        self.position_encoding = position_encoding

        self.embedding_layer = TokenPositionEmbedding(
            vocab_size=vocab_size,
            ctx_length=ctx_length,
            embedding_dim=embedding_dim,
            dropout=dropout,
            position_encoding=position_encoding,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    init_std=init_std,
                    dropout=dropout,
                    position_encoding=position_encoding,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(embedding_dim)
        self.W_output = nn.Linear(embedding_dim, vocab_size, bias=True)
        if tie_embeddings:
            self.W_output.weight = self.embedding_layer.token_embedding_layer.weight
        else:
            nn.init.normal_(self.W_output.weight, mean=0.0, std=init_std)
        nn.init.zeros_(self.W_output.bias)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding_layer(token_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_layer_norm(x)
        return self.W_output(x)
