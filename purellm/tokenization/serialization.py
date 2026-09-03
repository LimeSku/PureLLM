import json
from pathlib import Path
from typing import Any

from purellm.tokenization.bpe_tokenizer import BytePairTokenizer
from purellm.tokenization.char_tokenizer import CharacterTokenizer
from purellm.tokenization.tiktoken_tokenizer import TiktokenTokenizer
from purellm.tokenization.tokenizer import TextTokenizer


def tokenizer_from_dict(data: dict[str, Any]) -> TextTokenizer:
    tokenizer_type = data.get("type")
    if tokenizer_type == "character":
        return CharacterTokenizer.from_dict(data)
    if tokenizer_type == "byte_pair":
        return BytePairTokenizer.from_dict(data)
    if tokenizer_type == "tiktoken":
        return TiktokenTokenizer.from_dict(data)
    raise ValueError(f"unsupported tokenizer type: {tokenizer_type!r}")


def load_tokenizer(path: Path) -> TextTokenizer:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"invalid tokenizer file: {path}")
    return tokenizer_from_dict(data)


def save_tokenizer(path: Path, tokenizer: TextTokenizer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(tokenizer.to_dict(), indent=2),
        encoding="utf-8",
    )


def serialize_tokenizer(tokenizer: TextTokenizer) -> dict[str, Any]:
    tokenizer_config = tokenizer.to_dict()
    return {
        "tokenizer_type": tokenizer_config["type"],
        "tokenizer_config": tokenizer_config,
    }
