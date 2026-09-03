from typing import Any

import tiktoken


class TiktokenTokenizer:
    def __init__(self, encoding_name: str) -> None:
        self.name = encoding_name
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.encoding.max_token_value + 1

    def encode(self, text: str) -> list[int]:
        return self.encoding.encode(text)

    def decode(self, token_ids: list[int], errors: str = "strict") -> str:
        return self.encoding.decode(token_ids, errors=errors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "tiktoken",
            "version": 1,
            "encoding": self.name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TiktokenTokenizer":
        if data.get("type") != "tiktoken" or data.get("version") != 1:
            raise ValueError("invalid tiktoken tokenizer config")
        encoding_name = data.get("encoding")
        if not isinstance(encoding_name, str):
            raise ValueError("invalid tiktoken encoding name")
        return cls(encoding_name)
