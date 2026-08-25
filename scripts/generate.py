import argparse
from pathlib import Path

import torch

from purellm.torchgpt.checkpoint import load_model_checkpoint
from purellm.torchgpt.generation import generate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("prompt")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    device = select_device()
    checkpoint = load_model_checkpoint(
        args.checkpoint,
        device=device,
    )

    try:
        prompt_ids = checkpoint.tokenizer.encode(args.prompt)
    except KeyError as error:
        missing_character = error.args[0]
        raise ValueError(
            "prompt contains a character absent from the checkpoint tokenizer: "
            f"{missing_character!r}"
        ) from error

    torch.manual_seed(args.seed)
    generated_ids = generate(
        model=checkpoint.model,
        prompt_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=device,
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint} (step {checkpoint.step})")
    print(f"Prompt: {args.prompt!r}")
    print(checkpoint.tokenizer.decode(generated_ids, errors="replace"))


if __name__ == "__main__":
    main()
