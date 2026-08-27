from pathlib import Path

import numpy as np


class Adam:
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self, parameters_and_gradients) -> None:
        self.t += 1
        for param, grad in parameters_and_gradients:
            param_id = id(param)
            if param_id not in self.m:
                self.m[param_id] = np.zeros_like(param)
                self.v[param_id] = np.zeros_like(param)
            self.m[param_id] = self.beta1 * self.m[param_id] + (1 - self.beta1) * grad
            self.v[param_id] = self.beta2 * self.v[param_id] + (1 - self.beta2) * (
                grad**2
            )
            m_hat = self.m[param_id] / (1 - self.beta1**self.t)
            v_hat = self.v[param_id] / (1 - self.beta2**self.t)

            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

    def save_state(
        self,
        path: Path,
        named_parameters: dict[str, np.ndarray],
    ) -> None:
        state = {
            "t": np.array(self.t),
            "learning_rate": np.array(self.learning_rate),
            "beta1": np.array(self.beta1),
            "beta2": np.array(self.beta2),
            "eps": np.array(self.eps),
        }

        for name, param in named_parameters.items():
            param_id = id(param)

            if param_id in self.m:
                m = self.m[param_id]
                v = self.v[param_id]
            else:
                m = np.zeros_like(param)
                v = np.zeros_like(param)

            state[f"m.{name}"] = m
            state[f"v.{name}"] = v

        np.savez(path, **state)

    @classmethod
    def load_state(
        cls,
        path: Path,
        named_parameters: dict[str, np.ndarray],
    ) -> "Adam":
        with np.load(path, allow_pickle=False) as state:
            optimizer = cls(
                learning_rate=float(state["learning_rate"]),
                beta1=float(state["beta1"]),
                beta2=float(state["beta2"]),
                eps=float(state["eps"]),
            )
            optimizer.t = int(state["t"])

            for name, param in named_parameters.items():
                m = state[f"m.{name}"]
                v = state[f"v.{name}"]

                if m.shape != param.shape:
                    raise ValueError(
                        f"Adam first-moment shape mismatch for {name}: "
                        f"expected {param.shape}, got {m.shape}"
                    )

                if v.shape != param.shape:
                    raise ValueError(
                        f"Adam second-moment shape mismatch for {name}: "
                        f"expected {param.shape}, got {v.shape}"
                    )

                param_id = id(param)
                optimizer.m[param_id] = m.copy()
                optimizer.v[param_id] = v.copy()

        return optimizer
