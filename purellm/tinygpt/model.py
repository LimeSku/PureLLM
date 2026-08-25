import json
from pathlib import Path

import numpy as np

from purellm.tinygpt.embeddings import llmEmbeddingLayer
from purellm.tinygpt.transformer import LayerNorm, TransformerBlock


class TinyGPT:
    def __init__(
        self,
        vocab_size: int,
        ctx_length: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        hidden_dim: int,
        init_std: float = 0.02,
    ):
        self.vocab_size = vocab_size
        self.ctx_length = ctx_length
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.init_std = init_std

        self.embedding_layer = llmEmbeddingLayer(
            vocab_size=vocab_size,
            ctx_length=ctx_length,
            embedding_dim=embedding_dim,
        )
        self.blocks = [
            TransformerBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                init_std=init_std,
            )
            for _ in range(num_layers)
        ]

        self.final_layer_norm = LayerNorm(embedding_dim)
        self.W_output = np.random.normal(
            0.0,
            scale=init_std,
            size=(embedding_dim, vocab_size),
        )
        self.b_output = np.zeros(vocab_size)

    def __call__(self, token_ids: np.ndarray) -> np.ndarray:
        return self.forward(token_ids)

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        x = self.embedding_layer(token_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_layer_norm(x)
        self.last_hidden_state = x
        logits = x @ self.W_output + self.b_output
        return logits

    def backward(self, dlogits: np.ndarray) -> np.ndarray:
        hidden_flat = self.last_hidden_state.reshape(-1, self.embedding_dim)
        dlogits_flat = dlogits.reshape(-1, self.vocab_size)

        # self.dW_output = self.last_hidden_state.T @ dlogits
        # self.db_output = np.sum(dlogits, axis=0)
        self.dW_output = hidden_flat.T @ dlogits_flat
        self.db_output = np.sum(dlogits_flat, axis=0)
        dx = dlogits @ self.W_output.T
        dx = self.final_layer_norm.backward(dx)
        for block in reversed(self.blocks):
            dx = block.backward(dx)
        self.embedding_layer.backward(dx)
        return dx

    # def step(self, learning_rate: float) -> None:
    #     self.W_output -= learning_rate * self.dW_output
    #     self.b_output -= learning_rate * self.db_output
    #     self.final_layer_norm.step(learning_rate)
    #     for block in self.blocks:
    #         block.step(learning_rate)

    #     self.embedding_layer.step(learning_rate)

    def parameters_and_gradients(self):
        params = [
            (self.W_output, self.dW_output),
            (self.b_output, self.db_output),
        ]

        params += self.final_layer_norm.parameters_and_gradients()

        for block in self.blocks:
            params += block.parameters_and_gradients()

        params += self.embedding_layer.parameters_and_gradients()

        return params

    def named_parameters(self):
        params = {
            "W_output": self.W_output,
            "b_output": self.b_output,
        }

        params.update(self.final_layer_norm.named_parameters("final_layer_norm."))

        for i, block in enumerate(self.blocks):
            params.update(block.named_parameters(f"blocks.{i}."))

        params.update(self.embedding_layer.named_parameters("embedding_layer."))
        return params

    def get_config(self) -> dict[str, int | float]:
        return {
            "vocab_size": self.vocab_size,
            "ctx_length": self.ctx_length,
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "init_std": self.init_std,
        }

    def save_checkpoint(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

        config_path = path / "config.json"
        config_path.write_text(
            json.dumps(self.get_config(), indent=2),
            encoding="utf-8",
        )

        self.save_weights(path / "weights.npz")

    @classmethod
    def from_checkpoint(cls, path: Path) -> "TinyGPT":
        config_path = path / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))

        model = cls(**config)
        model.load_weights(path / "weights.npz")
        return model

    def save_weights(self, path) -> None:
        np.savez(path, **self.named_parameters())

    def load_weights(self, path) -> None:
        params = self.named_parameters()
        with np.load(path, allow_pickle=False) as weights:
            expected_names = set(params)
            checkpoint_names = set(weights.files)
            missing_names = expected_names - checkpoint_names
            if missing_names:
                names = ", ".join(sorted(missing_names))
                raise ValueError(f"Missing weights: {names}")
            unexpected_names = checkpoint_names - expected_names
            if unexpected_names:
                names = ", ".join(sorted(unexpected_names))
                raise ValueError(f"Missing weights: {names}")

            for name, param in params.items():
                weight = weights[name]
                if param.shape != weight.shape:
                    raise ValueError(
                        f"Shape mismatch for {name}: ",
                        f"expected {param.shape}, got {weight.shape}",
                    )
                param[...] = weight
