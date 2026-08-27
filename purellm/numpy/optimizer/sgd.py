class SGD:
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def step(self, parameters_and_gradients) -> None:
        for param, grad in parameters_and_gradients:
            param -= self.learning_rate * grad
