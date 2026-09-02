# PureLLM

PureLLM is an educational decoder-only language model implemented twice:
first in NumPy with manual backpropagation, then in PyTorch to explore modern
training and inference techniques without hiding the model behind a high-level
LLM framework.

## Highlights

- Character-level and BPE tokenizers
- TinyGPT implemented in NumPy, including manual backpropagation
- PyTorch implementation with RoPE, SwiGLU, and tied embeddings
- Native PyTorch scaled-dot-product attention
- Mixed-precision training on supported devices
- KV cache for autoregressive generation
- CPU, Apple Silicon/MPS, and CUDA support
- TOML experiment recipes, checkpoints, and training resume

## Quick start

Requirements: Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python scripts/train.py recipes/tinygpt/shakespeare_smoke.toml
```

Train the TinyStories model (fitted tokenizers are cached automatically):

```bash
uv run python scripts/train.py recipes/tinygpt/tiny_stories.toml
```

## Implementations

### NumPy

The NumPy implementation exposes the mechanics of a transformer explicitly:
forward pass, gradients, optimizer updates, and autoregressive generation.

It is designed for understanding and correctness, not training performance.

### PyTorch

The PyTorch implementation keeps the architecture visible while reusing
low-level optimized primitives where they improve performance or reliability,
including `torch.nn.functional.scaled_dot_product_attention`.

It is used to experiment with mixed precision, compilation, efficient
generation, and accelerator-specific performance.

## Example architecture

The TinyStories recipe currently trains an approximately 11M-parameter model:

| Component | Value |
| --- | ---: |
| Layers | 6 |
| Model dimension | 384 |
| Attention heads | 6 |
| Feed-forward dimension | 1,024 |
| Context length | 256 |
| Vocabulary | 1,024 |
| Positional encoding | RoPE |
| Activation | SwiGLU |

## Project structure

```text
purellm/
├── numpy/           # NumPy model and manual backpropagation
├── torchgpt/        # PyTorch model, training, and generation
└── tokenization/    # Character and BPE tokenizers

recipes/             # Reproducible TOML experiment configurations
scripts/             # Training, generation, and data preparation
runs/                # Checkpoints, tokenizer caches, and experiment outputs
```

## Scope

PureLLM is a learning and experimentation project, not a pretrained foundation
model or a replacement for Hugging Face Transformers.

Its purpose is to understand what LLM frameworks abstract away, then measure
the effect of selected PyTorch optimizations on a small, inspectable model.

See the [roadmap](ROADMAP.md) for planned experiments and improvements.
