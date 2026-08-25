from typing import Protocol


class TextTokenizer(Protocol):
    vocab_size: int

    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: list[int], errors: str = "strict") -> str: ...
