import numpy as np


class SequenceCrossEntropy:
    def __init__(self):
        self.probs = None
        self.targets = None

    def __call__(self, logits: np.ndarray, targets: np.ndarray) -> float:
        return self.forward(logits, targets)

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """Compute mean token cross-entropy using stable log-probabilities."""
        shifted_logits = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        sum_exp_logits = np.sum(exp_logits, axis=-1, keepdims=True)
        self.probs = exp_logits / sum_exp_logits
        self.targets = targets
        log_probs = shifted_logits - np.log(sum_exp_logits)
        log_probs_flat = log_probs.reshape(-1, log_probs.shape[-1])
        targets_flat = targets.reshape(-1)
        correct_log_probs = log_probs_flat[
            np.arange(len(targets_flat)),
            targets_flat,
        ]
        return float(-np.mean(correct_log_probs))

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
        so finally dL/dlogits = probs - one_hot(target), divided by the
        number of tokens for the mean loss.
        """
        dlogits = self.probs.copy()
        dlogits_flat = dlogits.reshape(-1, dlogits.shape[-1])
        targets_flat = self.targets.reshape(-1)
        n_tokens = len(targets_flat)

        dlogits_flat[np.arange(n_tokens), targets_flat] -= 1

        dlogits_flat /= n_tokens
        return dlogits
