import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal

Pretokenization = Literal["none", "whitespace"]
PRETOKEN_PATTERN = re.compile(r"\s+|[^\s]+")


def merge_pair(
    token_ids: list[int],
    pair: tuple[int, int],
    new_token_id: int,
) -> list[int]:
    merged = []
    index = 0
    while index < len(token_ids):
        if (
            index + 1 < len(token_ids)
            and (token_ids[index], token_ids[index + 1]) == pair
        ):
            merged.append(new_token_id)
            index += 2
        else:
            merged.append(token_ids[index])
            index += 1

    return merged


class BytePairTokenizer:
    def __init__(self) -> None:
        self.merges: list[tuple[int, int]] = []
        self.token_bytes: dict[int, bytes] = {
            token_id: bytes([token_id]) for token_id in range(256)
        }
        self._merge_ranks: dict[tuple[int, int], int] = {}
        self.vocab_size = 256
        self.pretokenization: Pretokenization = "whitespace"

    def fit(self, text: str, vocab_size: int) -> "BytePairTokenizer":
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")

        self.merges = []
        self.token_bytes = {token_id: bytes([token_id]) for token_id in range(256)}
        self._merge_ranks = {}
        self.vocab_size = 256
        self.pretokenization = "whitespace"

        piece_counts = Counter(
            match.group().encode("utf-8") for match in PRETOKEN_PATTERN.finditer(text)
        )
        sequences = Counter(
            {tuple(piece): count for piece, count in piece_counts.items()}
        )

        while self.vocab_size < vocab_size:
            pair_counts: Counter[tuple[int, int]] = Counter()
            for token_ids, count in sequences.items():
                for index in range(len(token_ids) - 1):
                    pair = (token_ids[index], token_ids[index + 1])
                    pair_counts[pair] += count

            if not pair_counts:
                break

            best_pair = min(
                pair_counts,
                key=lambda pair: (-pair_counts[pair], pair),
            )

            if pair_counts[best_pair] < 2:
                break

            new_token_id = self.vocab_size
            merged_sequences: Counter[tuple[int, ...]] = Counter()
            for token_ids, count in sequences.items():
                merged_token_ids = tuple(
                    merge_pair(
                        list(token_ids),
                        best_pair,
                        new_token_id,
                    )
                )
                merged_sequences[merged_token_ids] += count
            sequences = merged_sequences

            left, right = best_pair
            self.merges.append(best_pair)
            self._merge_ranks[best_pair] = new_token_id - 256
            self.token_bytes[new_token_id] = (
                self.token_bytes[left] + self.token_bytes[right]
            )
            self.vocab_size += 1
        return self

    def encode(self, text: str) -> list[int]:
        if self.pretokenization == "none":
            return self._encode_bytes(text.encode("utf-8"))

        cache: dict[bytes, list[int]] = {}
        token_ids = []
        for match in PRETOKEN_PATTERN.finditer(text):
            piece = match.group().encode("utf-8")
            encoded_piece = cache.get(piece)
            if encoded_piece is None:
                encoded_piece = self._encode_bytes(piece)
                cache[piece] = encoded_piece
            token_ids.extend(encoded_piece)
        return token_ids

    def _encode_bytes(self, raw_bytes: bytes) -> list[int]:
        token_ids = list(raw_bytes)

        while len(token_ids) > 1:
            ranked_pairs = (
                (self._merge_ranks[pair], pair)
                for pair in zip(token_ids, token_ids[1:])
                if pair in self._merge_ranks
            )
            best_rank_and_pair = min(ranked_pairs, default=None)
            if best_rank_and_pair is None:
                break

            rank, pair = best_rank_and_pair
            token_ids = merge_pair(
                token_ids,
                pair,
                new_token_id=256 + rank,
            )

        return token_ids

    def decode(self, token_ids: list[int], errors: str = "strict") -> str:
        raw_bytes = b"".join(self.token_bytes[token_id] for token_id in token_ids)
        return raw_bytes.decode("utf-8", errors=errors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "byte_pair",
            "version": 2,
            "pretokenization": self.pretokenization,
            "merges": [list(pair) for pair in self.merges],
        }

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "BytePairTokenizer":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"invalid tokenizer file: {path}")
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BytePairTokenizer":
        if data["type"] != "byte_pair":
            raise ValueError(f"Unsupported tokenizer type: {data['type']}")
        version = data.get("version")
        if version not in (1, 2):
            raise ValueError(f"unsupported byte-pair tokenizer version: {version!r}")

        tokenizer = cls()
        if version == 1:
            tokenizer.pretokenization = "none"
        else:
            pretokenization = data.get("pretokenization")
            if pretokenization not in ("none", "whitespace"):
                raise ValueError(f"unsupported pretokenization: {pretokenization!r}")
            tokenizer.pretokenization = pretokenization

        for serialized_pair in data["merges"]:
            if not isinstance(serialized_pair, list) or len(serialized_pair) != 2:
                raise ValueError(f"invalid byte-pair merge: {serialized_pair!r}")
            left, right = serialized_pair
            pair = (left, right)
            new_token_id = tokenizer.vocab_size

            tokenizer.merges.append(pair)
            tokenizer._merge_ranks[pair] = new_token_id - 256
            tokenizer.token_bytes[new_token_id] = (
                tokenizer.token_bytes[left] + tokenizer.token_bytes[right]
            )
            tokenizer.vocab_size += 1

        return tokenizer
