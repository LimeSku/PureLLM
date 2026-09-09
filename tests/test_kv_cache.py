import pytest
import torch

from purellm.torchgpt.model import TinyGPT


@pytest.mark.parametrize("normalization", ["layernorm", "rmsnorm"])
def test_kv_cache_matches_full_context(normalization: str) -> None:
    torch.manual_seed(0)
    tokens = torch.tensor([[1, 2, 3, 4, 5]])

    for position_encoding in ("learned", "rope"):
        model = TinyGPT(
            vocab_size=17,
            ctx_length=8,
            embedding_dim=8,
            num_heads=2,
            num_layers=2,
            hidden_dim=16,
            dropout=0.0,
            position_encoding=position_encoding,
            normalization=normalization,
        ).eval()
        norm_type = torch.nn.RMSNorm if normalization == "rmsnorm" else torch.nn.LayerNorm
        assert isinstance(model.final_layer_norm, norm_type)
        for block in model.blocks:
            assert isinstance(block.ln1, norm_type)
            assert isinstance(block.ln2, norm_type)

        with torch.no_grad():
            expected = [model(tokens[:, :end])[:, -1] for end in range(3, 6)]
            cached = [
                model(tokens[:, :3], use_cache=True)[:, -1],
                model(tokens[:, 3:4], use_cache=True)[:, -1],
                model(tokens[:, 4:5], use_cache=True)[:, -1],
            ]

        for cached_logits, expected_logits in zip(cached, expected, strict=True):
            torch.testing.assert_close(
                cached_logits,
                expected_logits,
                rtol=1e-5,
                atol=1e-6,
            )

        for block in model.blocks:
            assert block.attention.cache_k is not None
            assert block.attention.cache_v is not None
            assert block.attention.cache_k.shape == (1, 2, 5, 4)
            assert block.attention.cache_v.shape == (1, 2, 5, 4)

        model.reset_cache()
        for block in model.blocks:
            assert block.attention.cache_k is None
            assert block.attention.cache_v is None
