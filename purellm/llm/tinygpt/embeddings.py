import numpy as np


class Embedding:
    def __init__(self, num_embeddings: int, embedding_dim: int, init_std: float = 0.02):
        """
        vocab_size: number of tokens in the vocabulary
        embedding_dim: size of each token vector
        init_std=0.02 following GPT2 embeddings init
        """
        self.vocab_size = num_embeddings
        self.embedding_dim = embedding_dim

        self.weights = np.random.normal(
            0.0,
            scale=init_std,
            size=(num_embeddings, embedding_dim),
        )

    def __call__(self, token_ids: np.ndarray) -> np.ndarray:
        return self.forward(token_ids)

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        self.token_ids = token_ids
        return self.weights[token_ids]

    def backward(self, dout: np.ndarray) -> None:
        self.dweights = np.zeros_like(self.weights)
        np.add.at(
            self.dweights,
            self.token_ids.reshape(-1),
            dout.reshape(-1, self.embedding_dim),
        )

    def step(self, learning_rate: float) -> None:
        self.weights -= learning_rate * self.dweights

    def parameters_and_gradients(self):
        return [
            (self.weights, self.dweights),
        ]

    def named_parameters(self, prefix: str = ""):
        return {
            f"{prefix}weights": self.weights,
        }


class llmEmbeddingLayer:
    def __init__(self, vocab_size: int, ctx_length: int, embedding_dim: int):
        self.token_embedding_layer = Embedding(vocab_size, embedding_dim)
        self.pos_embedding_layer = Embedding(ctx_length, embedding_dim, init_std=0.01)

    def __call__(self, token_ids: np.ndarray) -> np.ndarray:
        return self.forward(token_ids)

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch_size, sequence_length)")

        batch_size, sequence_length = token_ids.shape
        positions = np.broadcast_to(
            np.arange(sequence_length),
            (batch_size, sequence_length),
        )
        token_embeddings = self.token_embedding_layer(token_ids)
        pos_embeddings = self.pos_embedding_layer(positions)
        return token_embeddings + pos_embeddings

    def backward(self, dout: np.ndarray) -> None:
        self.token_embedding_layer.backward(dout)
        self.pos_embedding_layer.backward(dout)

    def step(self, learning_rate: float) -> None:
        self.token_embedding_layer.step(learning_rate)
        self.pos_embedding_layer.step(learning_rate)

    def parameters_and_gradients(self):
        return (
            self.token_embedding_layer.parameters_and_gradients()
            + self.pos_embedding_layer.parameters_and_gradients()
        )

    def named_parameters(self, prefix: str = ""):
        params = {}
        params.update(
            self.token_embedding_layer.named_parameters(f"{prefix}token_embedding.")
        )
        params.update(
            self.pos_embedding_layer.named_parameters(f"{prefix}pos_embedding.")
        )
        return params
