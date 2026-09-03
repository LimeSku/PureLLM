import hashlib
from pathlib import Path

from purellm.config import TokenizerConfig
from purellm.tokenization.bpe_tokenizer import BytePairTokenizer
from purellm.tokenization.char_tokenizer import CharacterTokenizer
from purellm.tokenization.serialization import load_tokenizer, save_tokenizer
from purellm.tokenization.tiktoken_tokenizer import TiktokenTokenizer
from purellm.tokenization.tokenizer import TextTokenizer, TrainableTokenizer


def resolve_tokenizer(
    training_text: str,
    config: TokenizerConfig,
    cache_directory: Path,
) -> TextTokenizer:
    if config.type not in ("character", "bpe"):
        return TiktokenTokenizer(config.type)

    fit_text = training_text[: config.max_fit_characters]
    cache_path = _cache_path(cache_directory, fit_text, config)
    if cache_path.is_file():
        print(f"[tokenizer] cache hit | {cache_path}")
        tokenizer = load_tokenizer(cache_path)
        validate_tokenizer(tokenizer, config)
        return tokenizer

    tokenizer = _create_trainable_tokenizer(config).fit(fit_text)
    save_tokenizer(cache_path, tokenizer)
    print(f"[tokenizer] cache saved | {cache_path}")
    return tokenizer


def validate_tokenizer(
    tokenizer: TextTokenizer,
    config: TokenizerConfig,
) -> None:
    if tokenizer.name != config.type:
        raise ValueError(
            f"checkpoint tokenizer {tokenizer.name!r} does not match "
            f"configured tokenizer {config.type!r}"
        )


def _create_trainable_tokenizer(config: TokenizerConfig) -> TrainableTokenizer:
    if config.type == "character":
        return CharacterTokenizer()
    if config.vocab_size is None:
        raise ValueError("tokenizer.vocab_size is required for BPE")
    return BytePairTokenizer(config.vocab_size)


def _cache_path(
    cache_directory: Path,
    fit_text: str,
    config: TokenizerConfig,
) -> Path:
    corpus_hash = hashlib.sha256(fit_text.encode("utf-8")).hexdigest()[:12]
    if config.type == "character":
        filename = f"character-v1-{corpus_hash}.json"
    else:
        filename = f"bpe-v2-vocab{config.vocab_size}-{corpus_hash}.json"
    return cache_directory / filename
