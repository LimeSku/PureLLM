from collections.abc import Callable

import numpy as np
import pytest

from purellm.numpy.attention import (
    CausalSelfAttentionHead,
    MultiHeadCausalSelfAttention,
)
from purellm.numpy.embeddings import Embedding, llmEmbeddingLayer
from purellm.numpy.losses import SequenceCrossEntropy
from purellm.numpy.model import TinyGPT
from purellm.numpy.transformer import FeedForward, LayerNorm, TransformerBlock


def _assert_gradient(
    values: np.ndarray,
    analytical: np.ndarray,
    objective: Callable[[], float],
    name: str,
    step: float = 1e-6,
) -> None:
    numerical = np.empty_like(values)
    for index in np.ndindex(values.shape):
        original = values[index]
        try:
            values[index] = original + step
            plus = objective()
            values[index] = original - step
            minus = objective()
        finally:
            values[index] = original
        numerical[index] = (plus - minus) / (2 * step)

    np.testing.assert_allclose(
        analytical, numerical, rtol=2e-5, atol=2e-8, equal_nan=False, err_msg=name
    )


@pytest.mark.parametrize("gap", [None, 40.0, 1000.0])
def test_sequence_cross_entropy_gradient(gap: float | None) -> None:
    rng = np.random.default_rng(7)
    logits = rng.normal(size=(2, 3, 5))
    targets = np.array([[1, 2, 1], [3, 0, 4]])
    target_indices = (*np.indices(targets.shape), targets)
    if gap is not None:
        logits.fill(0.0)
        logits[target_indices] = -gap

    loss = SequenceCrossEntropy()
    value = loss(logits, targets)
    analytical = loss.backward().copy()
    expected = np.mean(np.logaddexp.reduce(logits, axis=-1) - logits[target_indices])
    assert np.isfinite(value)
    assert np.all(np.isfinite(analytical))
    _assert_gradient(
        logits,
        analytical,
        lambda: loss(logits, targets),
        "loss logits",
        step=1e-6 if gap is None else 1e-5,
    )
    np.testing.assert_allclose(value, expected, rtol=1e-12, atol=1e-12)


def test_layer_gradients() -> None:
    np.random.seed(7)
    rng = np.random.default_rng(7)
    tokens = np.array([[1, 1, 2], [2, 1, 1]])
    inputs = rng.normal(size=(2, 3, 4))
    layers = [
        Embedding(5, 4),
        llmEmbeddingLayer(5, 3, 4),
        CausalSelfAttentionHead(4, 2, init_std=0.2),
        MultiHeadCausalSelfAttention(4, 2, init_std=0.2),
        FeedForward(4, 5, init_std=0.2),
        LayerNorm(4),
        TransformerBlock(4, 2, 5, init_std=0.2),
    ]
    for layer in layers:
        name = type(layer).__name__
        is_embedding = isinstance(layer, (Embedding, llmEmbeddingLayer))
        x = tokens if is_embedding else inputs.copy()
        dout = rng.normal(size=layer(x).shape)
        dx = layer.backward(dout)
        gradients = [(p, g.copy()) for p, g in layer.parameters_and_gradients()]
        names = {id(p): key for key, p in layer.named_parameters().items()}

        def objective() -> float:
            return float(np.sum(layer(x) * dout))

        if not is_embedding:
            _assert_gradient(x, dx.copy(), objective, f"{name} input")
        for parameter, gradient in gradients:
            _assert_gradient(
                parameter, gradient, objective, f"{name}.{names[id(parameter)]}"
            )


def test_tinygpt_parameter_gradients() -> None:
    np.random.seed(7)
    model = TinyGPT(
        vocab_size=5,
        ctx_length=3,
        embedding_dim=4,
        num_heads=2,
        num_layers=2,
        hidden_dim=5,
        init_std=0.2,
    )
    tokens = np.array([[1, 1, 2], [2, 1, 1]])
    targets = np.array([[2, 2, 1], [1, 2, 2]])
    loss = SequenceCrossEntropy()
    loss(model(tokens), targets)
    model.backward(loss.backward())
    gradients = [(p, g.copy()) for p, g in model.parameters_and_gradients()]
    names = {id(p): name for name, p in model.named_parameters().items()}
    for parameter, gradient in gradients:
        _assert_gradient(
            parameter,
            gradient,
            lambda: loss(model(tokens), targets),
            names[id(parameter)],
        )
