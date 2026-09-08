from pathlib import Path

from purellm.tokenization import BytePairTokenizer, CharacterTokenizer
from purellm.tokenization.serialization import load_tokenizer, save_tokenizer


def test_character_tokenizer_round_trip_after_serialization(tmp_path: Path) -> None:
    text = "cab 🙂"
    tokenizer = CharacterTokenizer().fit(text)
    token_ids = tokenizer.encode(text)

    assert tokenizer.decode(token_ids) == text

    path = tmp_path / "character.json"
    save_tokenizer(path, tokenizer)
    restored = load_tokenizer(path)

    assert restored.vocab_size == tokenizer.vocab_size
    assert restored.encode(text) == token_ids
    assert restored.decode(token_ids) == text


def test_bpe_tokenizer_round_trip_after_serialization(tmp_path: Path) -> None:
    text = "hello hello 🙂🙂\nhello"
    tokenizer = BytePairTokenizer(270).fit(text)
    token_ids = tokenizer.encode(text)

    assert tokenizer.vocab_size > 256
    assert len(token_ids) < len(text.encode())
    assert tokenizer.decode(token_ids) == text

    path = tmp_path / "bpe.json"
    save_tokenizer(path, tokenizer)
    restored = load_tokenizer(path)

    assert restored.vocab_size == tokenizer.vocab_size
    assert restored.encode(text) == token_ids
    assert restored.decode(token_ids) == text
