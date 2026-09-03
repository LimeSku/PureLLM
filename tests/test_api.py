from pathlib import Path
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient

from purellm.api import MAX_NEW_TOKENS, create_app
from purellm.tokenization import CharacterTokenizer, serialize_tokenizer
from purellm.torchgpt.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    load_model_checkpoint,
)
from purellm.torchgpt.model import TinyGPT


def test_inference_api(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.pt"
    tokenizer = CharacterTokenizer().fit(" ab")
    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        ctx_length=4,
        embedding_dim=4,
        num_heads=1,
        num_layers=1,
        hidden_dim=8,
    )
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_config": {
                "vocab_size": model.vocab_size,
                "ctx_length": model.ctx_length,
                "embedding_dim": model.embedding_dim,
                "num_heads": model.num_heads,
                "num_layers": model.num_layers,
                "hidden_dim": model.hidden_dim,
            },
            "model_state_dict": model.state_dict(),
            **serialize_tokenizer(tokenizer),
            "step": 12,
            "best_validation_loss": 1.0,
        },
        checkpoint_path,
    )

    with patch(
        "purellm.api.load_model_checkpoint",
        wraps=load_model_checkpoint,
    ) as loader:
        with TestClient(
            create_app(checkpoint_path, device=torch.device("cpu"))
        ) as client:
            health = client.get("/health")
            generated = client.post(
                "/generate",
                json={"prompt": "a", "max_new_tokens": 2, "temperature": 1.0},
            )
            invalid = client.post(
                "/generate",
                json={"prompt": "a", "max_new_tokens": MAX_NEW_TOKENS + 1},
            )
            openapi = client.get("/openapi.json")

    assert loader.call_count == 1
    assert health.json() == {"status": "ok", "device": "cpu", "checkpoint_step": 12}
    assert generated.status_code == 200
    assert generated.json()["text"].startswith("a")
    assert generated.json()["generated_tokens"] == 2
    assert generated.json()["latency_ms"] > 0
    assert generated.json()["tokens_per_second"] > 0
    assert invalid.status_code == 422
    assert {"/health", "/generate"} <= openapi.json()["paths"].keys()
