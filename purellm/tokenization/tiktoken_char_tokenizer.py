import string

import tiktoken

DEFAULT_CHARACTERS = string.ascii_letters + string.digits + string.punctuation + " \n"


class TiktokenGPT2:
    def __init__(self) -> None:
        # self.char_to_id: dict[str, int] | None = None
        # self.id_to_char: dict[int, str] | None = None

        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.tokenizer.max_token_value + 1

    def fit(self, text: str = DEFAULT_CHARACTERS) -> "TiktokenGPT2":
        # unique_chars = sorted(set(text))
        # self.char_to_id = {char: i for i, char in enumerate(unique_chars)}
        # self.id_to_char = {i: char for i, char in enumerate(unique_chars)}
        # self.vocab_size = len(unique_chars)
        return self

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int], errors: str = "strict") -> str:
        return self.tokenizer.decode(ids)
        # return "".join(self.id_to_char[token_id] for token_id in ids)


if __name__ == "__main__":
    tokenizer = TiktokenGPT2()
    tokenizer.fit()
