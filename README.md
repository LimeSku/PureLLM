# PureLLM

Educational language-model implementations built from scratch.

## What is included

- Character and byte-pair tokenizers
- A NumPy TinyGPT implementation with manual backpropagation
- A PyTorch TinyGPT implementation with CUDA, MPS, and CPU support
- Training, generation, checkpointing, RoPE, SwiGLU, and weight tying
- Tiny Shakespeare, TinyStories, and Discord corpus helpers

## Setup

```bash
uv sync
```

## Run

Validate the example experiment configuration:

```bash
uv run python -m purellm.config recipes/tinygpt/shakespeare.toml
```

Train on the included Tiny Shakespeare corpus. Model, data, and training settings
come from the TOML file:

```bash
uv run python scripts/train.py recipes/tinygpt/shakespeare.toml
```

Each fresh launch creates the next numbered run directory, such as
`runs/tinygpt-shakespeare/1/` and `runs/tinygpt-shakespeare/2/`, and saves the
recipe as `config.toml` inside it. Resuming from a checkpoint keeps writing to
its existing run directory.

Available recipes cover the main implemented paths:

| Recipe | Purpose |
| --- | --- |
| `shakespeare.toml` | Character tokenizer, RoPE, tied embeddings |
| `shakespeare_smoke.toml` | Short local check, learned positions, untied embeddings |
| `tiny_stories.toml` | BPE and separate train/validation files |
| `discord.toml` | BPE on a converted conversational corpus |

Generate text from a checkpoint:

```bash
uv run python scripts/generate.py \
  runs/tinygpt-shakespeare/1/checkpoints/best.pt "ROMEO:"
```

The pedagogical NumPy implementation has a separate training example:

```bash
uv run python examples/numpy/train.py shakespeare
```

## Optional corpora

TinyStories is downloaded rather than committed because it uses about 2.25 GB:

```bash
uv run python scripts/download_dataset.py tiny-stories
```

Then train with `recipes/tinygpt/tiny_stories.toml`.

Discord exports can be converted into a private next-token corpus:

```bash
uv run python scripts/prepare_discord_dataset.py path/to/discord-exports \
  --output datasets/discord/input.txt
```

Then train with `recipes/tinygpt/discord.toml`.
See the [conversation training guide](chat_finetuning.md) for the full workflow.

Generated datasets and checkpoints are ignored by Git.

## Structure

```text
datasets/tiny_shakespeare/   Small tracked corpus
examples/numpy/              Pedagogical NumPy training example
purellm/                     Models, tokenizers, config, and training primitives
recipes/tinygpt/             Reproducible experiment configurations
scripts/                     Training, generation, and corpus commands
```
