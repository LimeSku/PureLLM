from typing import Any, Protocol, Self


class TextTokenizer(Protocol):
    name: str
    vocab_size: int

    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: list[int], errors: str = "strict") -> str: ...

    def to_dict(self) -> dict[str, Any]: ...


class TrainableTokenizer(TextTokenizer, Protocol):
    def fit(self, text: str) -> Self: ...
