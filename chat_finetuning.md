# Train TinyGPT on conversations

In PureLLM, this procedure trains a new TinyGPT model on a conversation corpus.
It does not fine-tune a pretrained model.

## Discord

Only use conversations that you have permission to process. Exports may contain
private data: do not share or commit them. The `datasets/discord/` directory is
already ignored by Git.

### 1. Export the conversations

1. Download the stable GUI or CLI release of
   [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter).
2. Prefer authenticating with a bot that can access the relevant channels. The
   project warns that automating a user account violates Discord's terms of
   service.
3. Export the channels as **JSON** into a directory such as
   `datasets/discord/raw/`. A single file or multiple subdirectories both work.

### 2. Prepare the corpus

From the repository root:

```bash
uv sync
uv run python scripts/prepare_discord_dataset.py datasets/discord/raw \
  --output datasets/discord/input.txt
```

By default, the script anonymizes speakers, replaces URLs, excludes bots and
system messages, and produces:

- `datasets/discord/input.txt`: the training corpus;
- `datasets/discord/speakers.json`: the private speaker alias mapping;
- `datasets/discord/stats.json`: conversion statistics.

Check `stats.json` and review `input.txt` before starting training. Display the
available options with:

```bash
uv run python scripts/prepare_discord_dataset.py --help
```

### 3. Train with the Discord recipe

The `recipes/tinygpt/discord.toml` recipe reads the corpus created in the
previous step. Validate it, then start training:

```bash
uv run python -m purellm.config recipes/tinygpt/discord.toml
uv run python scripts/train.py recipes/tinygpt/discord.toml
```

Checkpoints are written to a numbered directory such as
`runs/tinygpt-discord/1/checkpoints/`. To test the best checkpoint:

```bash
uv run python scripts/generate.py \
  runs/tinygpt-discord/1/checkpoints/best.pt "<USER_0001>"
```
