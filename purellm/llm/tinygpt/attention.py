import numpy as np


class CausalSelfAttentionHead:
    def __init__(self, embedding_dim: int, head_dim: int, init_std: float = 0.02):
        self.embedding_dim = embedding_dim
        self.head_dim = head_dim

        self.W_query = np.random.normal(0.0, init_std, size=(embedding_dim, head_dim))
        self.W_key = np.random.normal(0.0, init_std, size=(embedding_dim, head_dim))
        self.W_value = np.random.normal(0.0, init_std, size=(embedding_dim, head_dim))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        Q = x @ self.W_query
        K = x @ self.W_key
        V = x @ self.W_value

        attention_scores = Q @ K.swapaxes(-2, -1)
        attention_scores = attention_scores / np.sqrt(self.head_dim)

        sequence_length = x.shape[-2]
        mask = np.triu(
            np.ones((sequence_length, sequence_length), dtype=bool),
            k=1,
        )
        attention_scores = np.where(mask, -np.inf, attention_scores)
        attention_weights = self._softmax(attention_scores)

        self.x = x
        self.Q = Q
        self.K = K
        self.V = V
        self.scores = attention_scores
        self.mask = mask
        self.weights = attention_weights
        return attention_weights @ V

    def backward(self, dout):
        dweights = dout @ self.V.swapaxes(-2, -1)
        dV = self.weights.swapaxes(-2, -1) @ dout
        dscores = self.weights * (
            dweights - np.sum(dweights * self.weights, axis=-1, keepdims=True)
        )
        dscores = np.where(self.mask, 0.0, dscores)
        dscores = dscores / np.sqrt(self.head_dim)
        dQ = dscores @ self.K
        dK = dscores.swapaxes(-2, -1) @ self.Q
        x_flat = self.x.reshape(-1, self.embedding_dim)
        self.dW_query = x_flat.T @ dQ.reshape(-1, self.head_dim)
        self.dW_key = x_flat.T @ dK.reshape(-1, self.head_dim)
        self.dW_value = x_flat.T @ dV.reshape(-1, self.head_dim)

        # dX = sum(gradients from all paths)
        dx = dQ @ self.W_query.T + dK @ self.W_key.T + dV @ self.W_value.T

        return dx

    def step(self, learning_rate: float) -> None:
        self.W_query -= learning_rate * self.dW_query
        self.W_key -= learning_rate * self.dW_key
        self.W_value -= learning_rate * self.dW_value

    def parameters_and_gradients(self):
        return [
            (self.W_query, self.dW_query),
            (self.W_key, self.dW_key),
            (self.W_value, self.dW_value),
        ]

    def named_parameters(self, prefix: str = ""):
        return {
            f"{prefix}W_query": self.W_query,
            f"{prefix}W_key": self.W_key,
            f"{prefix}W_value": self.W_value,
        }

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class MultiHeadCausalSelfAttention:
    def __init__(self, embedding_dim: int, num_heads: int, init_std: float = 0.02):
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.heads = [
            CausalSelfAttentionHead(
                embedding_dim=embedding_dim, head_dim=self.head_dim, init_std=init_std
            )
            for _ in range(num_heads)
        ]
        self.W_output = np.random.normal(
            0.0,
            init_std,
            size=(embedding_dim, embedding_dim),
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        head_outputs = [head(x) for head in self.heads]
        concatenated = np.concatenate(head_outputs, axis=-1)
        self.concatenated = concatenated
        return concatenated @ self.W_output

    def backward(self, dout):
        concatenated_flat = self.concatenated.reshape(-1, self.embedding_dim)
        dout_flat = dout.reshape(-1, self.embedding_dim)
        self.dW_output = concatenated_flat.T @ dout_flat
        dconcatenated = dout @ self.W_output.T
        head_grads = np.split(dconcatenated, self.num_heads, axis=-1)
        dx = sum(
            head.backward(head_grad) for head, head_grad in zip(self.heads, head_grads)
        )
        return dx

    def step(self, learning_rate: float) -> None:
        self.W_output -= learning_rate * self.dW_output
        for head in self.heads:
            head.step(learning_rate)

    def parameters_and_gradients(self):
        params = [
            (self.W_output, self.dW_output),
        ]

        for head in self.heads:
            params += head.parameters_and_gradients()

        return params

    def named_parameters(self, prefix: str = ""):
        params = {
            f"{prefix}W_output": self.W_output,
        }

        for index, head in enumerate(self.heads):
            params.update(head.named_parameters(f"{prefix}heads.{index}."))

        return params
