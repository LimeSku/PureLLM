from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from purellm.config import DataConfig


class LLMDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        self.token_ids = torch.tensor(
            tokenizer.encode(txt),
            dtype=torch.long,
        )
        self.stride = stride
        self.max_length = max_length
        # Use a sliding window to chunk the book into overlapping sequences of max_length
        # for i in range(0, len(token_ids) - max_length, stride):
        # input_chunk = token_ids[i : i + max_length]
        # target_chunk = token_ids[i + 1 : i + max_length + 1]
        # self.input_ids.append(torch.tensor(input_chunk))
        # self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(range(0, len(self.token_ids) - self.max_length, self.stride))
        # return len(self.input_ids)

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.token_ids[start : start + self.max_length]
        y = self.token_ids[start + 1 : start + self.max_length + 1]
        return x, y
        # return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
    txt,
    tokenizer,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    # here we suppose the tokenizer is already fitted
    # Create dataset
    dataset = LLMDataset(txt, tokenizer, max_length, stride)
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader


def read_text(path: Path, max_characters: int | None = None) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"dataset file not found: {path}")

    with path.open(encoding="utf-8") as file:
        return file.read(max_characters)


def load_texts(config: DataConfig) -> tuple[str, str]:
    if config.validation_path is not None:
        training_text = read_text(config.train_path, config.max_train_characters)
        validation_text = read_text(
            config.validation_path,
            config.max_validation_characters,
        )
    else:
        text = read_text(config.train_path)
        validation_fraction = config.validation_fraction
        if validation_fraction is None:
            raise ValueError(
                "data.validation_fraction is required without validation_path"
            )

        split_index = int(len(text) * (1 - validation_fraction))
        training_text = text[:split_index]
        validation_text = text[split_index:]
        if config.max_train_characters is not None:
            training_text = training_text[: config.max_train_characters]
        if config.max_validation_characters is not None:
            validation_text = validation_text[: config.max_validation_characters]

    if not training_text or not validation_text:
        raise ValueError("training and validation data must not be empty")
    return training_text, validation_text
