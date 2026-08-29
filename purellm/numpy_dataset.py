from purellm.tokenization import CharacterTokenizer


class llmDataset:
    def __init__(self, token_ids: list[int], ctx_length: int):
        if len(token_ids) <= ctx_length:
            raise ValueError("token_ids must contain more tokens than ctx_length")
        self.token_ids = token_ids
        self.ctx_length = ctx_length

    def __len__(self) -> int:
        return len(self.token_ids) - self.ctx_length

    def __getitem__(self, index):
        x = self.token_ids[index : index + self.ctx_length]
        y = self.token_ids[index + 1 : index + self.ctx_length + 1]
        return x, y

    @classmethod
    def from_text(
        cls,
        text: str,
        tokenizer: CharacterTokenizer,
        ctx_length: int,
    ) -> "llmDataset":
        """
        tokenizer: must be fit, will raise otherwise
        """
        token_ids = tokenizer.encode(text)
        return cls(token_ids=token_ids, ctx_length=ctx_length)
