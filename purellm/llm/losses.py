import numpy as np


class SequenceCrossEntropy:
    def __init__(self):
        self.probs = None
        self.targets = None
        self.eps = 1e-12

    def __call__(self, logits: np.ndarray, targets: np.ndarray) -> float:
        return self.forward(logits, targets)

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """
        1. compute probs of each class by softmax on logits
        2. identify the predicted proba for each actual target
        3. take the - mean(log(proba + epsilon)) to get value close to 0 when proba is ok, very high otherwise
        """
        self.probs = self._softmax(logits)
        self.targets = targets
        probs_flat = self.probs.reshape(-1, self.probs.shape[-1])
        targets_flat = targets.reshape(-1)
        # correct_probs = self.probs[np.arange(len(logits)), targets]
        correct_probs = probs_flat[
            np.arange(len(targets_flat)),
            targets_flat,
        ]
        return -np.mean(np.log(correct_probs + self.eps))

    def backward(self) -> np.ndarray:
        """
        dL/dlogits
        derivative (crossentropy + softmax) can be simplified into:
        dlogits = probs - onehot(targets)

        z = logits
        for class i and correct class y:
        p_i = exp(z_i) / sum_j exp(z_j)

        L = -log(p_y)
        L = -log(exp(z_y) / sum_j exp(z_j))
        L = -[log(exp(z_y)) - log(sum_j exp(z_j))]
        L = -z_y + log(sum_j exp(z_j))

        => derivative wrt k logit z_k:
        dL/dz_k = d/dz_k [-z_y + log(sum_j exp(z_j))]
        first term: -one_hot(y)_k
        second term: exp(z_k) / sum_j exp(z_j) <=> p_k
        so finally dL/dlogits = probs - one_hot(target)
        """
        # n_tokens = self.probs.shape[0]
        dlogits = self.probs.copy()
        dlogits_flat = dlogits.reshape(-1, dlogits.shape[-1])
        targets_flat = self.targets.reshape(-1)
        n_tokens = len(targets_flat)

        # dlogits[np.arange(n_tokens), self.targets] -= 1
        # dlogits /= n_tokens
        dlogits_flat[np.arange(n_tokens), targets_flat] -= 1

        dlogits_flat /= n_tokens
        return dlogits

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
