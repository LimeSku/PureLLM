import numpy as np

from purellm.llm.tinygpt.model import TinyGPT


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def generate(
    model: TinyGPT,
    token_ids: list[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> list[int]:
    generated = list(token_ids)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive")
    for _ in range(max_new_tokens):
        context = generated[-model.ctx_length :]
        logits = model(np.asarray(context)[None, :])

        next_token_logits = logits[0, -1] / temperature
        if top_k is not None:
            k = min(top_k, len(next_token_logits))
            top_indices = np.argpartition(next_token_logits, -k)[-k:]
            masked_logits = np.full_like(next_token_logits, -np.inf)
            masked_logits[top_indices] = next_token_logits[top_indices]
            next_token_logits = masked_logits
        probs = softmax(next_token_logits)
        next_token_id = int(np.random.choice(len(probs), p=probs))
        generated.append(next_token_id)

    return generated
