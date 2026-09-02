import string
from typing import Any

DEFAULT_CHARACTERS = string.ascii_letters + string.digits + string.punctuation + " \n"


class CharacterTokenizer:
    name = "character"

    def __init__(self) -> None:
        self.char_to_id: dict[str, int] | None = None
        self.id_to_char: dict[int, str] | None = None
        self.vocab_size = 0

    def fit(self, text: str = DEFAULT_CHARACTERS) -> "CharacterTokenizer":
        unique_chars = sorted(set(text))
        self.char_to_id = {char: i for i, char in enumerate(unique_chars)}
        self.id_to_char = {i: char for i, char in enumerate(unique_chars)}
        self.vocab_size = len(unique_chars)
        return self

    def encode(self, text: str) -> list[int]:
        if self.char_to_id is None:
            raise ValueError("Tokenizer not trained yet")
        return [self.char_to_id[character] for character in text]

    def decode(self, ids: list[int], errors: str = "strict") -> str:
        if self.id_to_char is None:
            raise ValueError("Tokenizer not trained yet")
        return "".join(self.id_to_char[token_id] for token_id in ids)

    def to_dict(self) -> dict[str, Any]:
        if self.id_to_char is None:
            raise ValueError("tokenizer must be fitted before serialization")
        return {
            "type": "character",
            "version": 1,
            "characters": "".join(
                self.id_to_char[token_id] for token_id in range(self.vocab_size)
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CharacterTokenizer":
        if data.get("type") != "character" or data.get("version") != 1:
            raise ValueError("invalid character tokenizer config")
        characters = data.get("characters")
        if not isinstance(characters, str):
            raise ValueError("invalid character tokenizer characters")
        return cls().fit(characters)


if __name__ == "__main__":
    tokenizer = CharacterTokenizer()
    tokenizer.fit()
    sentence = "Hi, im doing nonsense !"
    tokenized = tokenizer.encode(sentence)
    print(f"Initial sentence: {sentence}")
    print(f"Tokenized sentence: {tokenized}")
    print(f"Decoded sentence: {tokenizer.decode(tokenized)}")
