import argparse
from pathlib import Path

import numpy as np

from purellm.generation import generate
from purellm.losses import SequenceCrossEntropy
from purellm.tinygpt_numpy.model import TinyGPT
from purellm.tokenization import BytePairTokenizer
from purellm.training import train_language_model_batch
from purellm.optimizer.adam import Adam
from purellm.optimizer.sgd import SGD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        choices=["shakespeare", "toy"],
        nargs="?",
        default="shakespeare",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("datasets/tiny_shakespeare/input.txt"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=20_000)
    parser.add_argument("--ctx-length", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--prompt")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--save-checkpoint", type=Path)
    parser.add_argument("--load-checkpoint", type=Path)
    return parser.parse_args()


def evaluate_language_model(
    model: TinyGPT,
    token_ids: list[int],
    starts: np.ndarray,
) -> float:
    loss_fn = SequenceCrossEntropy()
    x_batch = np.asarray([
        token_ids[start : start + model.ctx_length] for start in starts
    ])
    y_batch = np.asarray([
        token_ids[start + 1 : start + model.ctx_length + 1] for start in starts
    ])
    logits = model(x_batch)
    return float(loss_fn(logits, y_batch))


def main() -> None:
    args = parse_args()
    np.random.seed(42)
    rng = np.random.default_rng(42)

    if args.dataset == "toy":
        text = "hello world hello world hello world"
        train_text = text
        validation_text = text
        ctx_length = args.ctx_length if args.ctx_length is not None else 4
        steps = args.steps
        learning_rate = args.learning_rate if args.learning_rate is not None else 0.1
        log_every = args.log_every
        prompt = args.prompt if args.prompt is not None else "hell"
        max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else 24
        embedding_dim = 8
        num_heads = 2
        num_layers = 2
        hidden_dim = 32
        optimizer = SGD(learning_rate=learning_rate)
    else:
        text = args.path.read_text()[: args.max_chars]
        split_index = int(len(text) * 0.9)
        train_text = text[:split_index]
        validation_text = text[split_index:]
        ctx_length = args.ctx_length if args.ctx_length is not None else 32
        steps = args.steps if args.steps is not None else 1_000
        learning_rate = args.learning_rate if args.learning_rate is not None else 0.001
        log_every = args.log_every if args.log_every is not None else 100
        prompt = args.prompt if args.prompt is not None else "ROMEO:"
        max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else 200
        embedding_dim = 128
        num_heads = 4
        num_layers = 4
        hidden_dim = 512
        optimizer = Adam(learning_rate=learning_rate)

    if args.load_checkpoint is not None:
        tokenizer = BytePairTokenizer.load(args.load_checkpoint / "tokenizer.json")
        model = TinyGPT.from_checkpoint(args.load_checkpoint)
        if isinstance(optimizer, Adam):
            optimizer = Adam.load_state(
                args.load_checkpoint / "optimizer.npz",
                model.named_parameters(),
            )
            if args.learning_rate is not None:
                optimizer.learning_rate = args.learning_rate
        if args.ctx_length is not None and args.ctx_length != model.ctx_length:
            raise ValueError(
                "ctx_length does not match the loaded checkpoint: "
                f"expected {model.ctx_length}, got {args.ctx_length}"
            )
        if tokenizer.vocab_size != model.vocab_size:
            raise ValueError(
                "Tokenizer vocabulary size does not match the loaded checkpoint: "
                f"expected {model.vocab_size}, got {tokenizer.vocab_size}"
            )
        ctx_length = model.ctx_length
    else:
        tokenizer = BytePairTokenizer().fit(train_text, vocab_size=512)

        model = TinyGPT(
            vocab_size=tokenizer.vocab_size,
            ctx_length=ctx_length,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
        )
    token_ids = tokenizer.encode(train_text)
    validation_token_ids = tokenizer.encode(validation_text)

    if len(token_ids) <= ctx_length:
        raise ValueError("Training text must be longer than ctx_length")
    if args.dataset == "toy":
        if args.steps is None:
            steps = 200 * (len(train_text) - ctx_length)
        if args.log_every is None:
            log_every = max(1, len(train_text) - ctx_length)

    eval_rng = np.random.default_rng(123)
    validation_max_start = len(validation_token_ids) - ctx_length
    validation_starts = eval_rng.integers(
        0,
        validation_max_start,
        size=min(200, validation_max_start),
    )
    loss_fn = SequenceCrossEntropy()

    print(f"Dataset: {args.dataset}")
    print(f"Dataset characters: {len(text)}")
    print(f"Training characters: {len(train_text)}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Context length: {ctx_length}")
    print()

    running_loss = 0.0
    max_start = len(token_ids) - ctx_length
    for step in range(steps):
        starts = rng.integers(
            0,
            max_start,
            size=args.batch_size,
        )
        x_batch = np.asarray([
            token_ids[start : start + ctx_length] for start in starts
        ])
        y_batch = np.asarray([
            token_ids[start + 1 : start + ctx_length + 1] for start in starts
        ])
        loss = train_language_model_batch(
            model,
            loss_fn,
            optimizer,
            x_batch,
            y_batch,
            max_grad_norm=args.max_grad_norm,
        )

        running_loss += loss
        if step == 0 or (step + 1) % log_every == 0:
            train_loss = running_loss / min(log_every, step + 1)
            validation_loss = evaluate_language_model(
                model,
                validation_token_ids,
                validation_starts,
            )
            print(
                f"Step {step + 1:>5}/{steps} "
                f"- train loss: {train_loss:.4f} "
                f"- validation loss: {validation_loss:.4f}"
            )
            running_loss = 0.0

    if args.save_checkpoint is not None:
        model.save_checkpoint(args.save_checkpoint)
        tokenizer.save(args.save_checkpoint / "tokenizer.json")
        if isinstance(optimizer, Adam):
            optimizer.save_state(
                args.save_checkpoint / "optimizer.npz",
                model.named_parameters(),
            )
    prompt_ids = tokenizer.encode(prompt)
    generated_ids = generate(
        model=model,
        token_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print()
    print(f"Prompt: {prompt!r}")
    print(tokenizer.decode(generated_ids, errors="replace"))


if __name__ == "__main__":
    main()
