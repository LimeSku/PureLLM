import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

CONVERSATION_START = "<CONVERSATION>"
CONVERSATION_END = "<END_CONVERSATION>"
INCLUDED_MESSAGE_TYPES = {"Default", "Reply"}
RESERVED_SPEAKER_LABELS = {
    "AUDIO",
    "CHANNEL",
    "CONVERSATION",
    "EMBED",
    "END_CONVERSATION",
    "FILE",
    "IMAGE",
    "REPLY",
    "ROLE",
    "STICKER",
    "TIMESTAMP",
    "URL",
    "USER_UNKNOWN",
    "VIDEO",
}
IMAGE_EXTENSIONS = {".avif", ".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav"}

URL_PATTERN = re.compile(r"https?://[^\s<]+")
USER_MENTION_PATTERN = re.compile(r"<@!?(\d+)>")
ROLE_MENTION_PATTERN = re.compile(r"<@&\d+>")
CHANNEL_MENTION_PATTERN = re.compile(r"<#\d+>")
CUSTOM_EMOJI_PATTERN = re.compile(r"<a?:([A-Za-z0-9_]+):\d+>")
DISCORD_TIMESTAMP_PATTERN = re.compile(r"<t:\d+(?::[A-Za-z])?>")


@dataclass(frozen=True)
class DiscordMessage:
    id: str
    channel_id: str
    timestamp: datetime
    timestamp_edited: datetime | None
    message_type: str
    author_id: str
    author_name: str
    author_nickname: str
    is_bot: bool
    content: str
    reference_message_id: str | None
    source_path: Path

    def duplicate_signature(self) -> tuple[Any, ...]:
        return (
            self.channel_id,
            self.timestamp,
            self.message_type,
            self.author_id,
            self.content,
            self.reference_message_id,
        )


@dataclass(frozen=True)
class Conversation:
    channel_id: str
    messages: tuple[DiscordMessage, ...]

    @property
    def started_at(self) -> datetime:
        return self.messages[0].timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert DiscordChatExporter JSON exports into a plain next-token "
            "training corpus."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="DiscordChatExporter JSON files or directories containing them",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/discord/input.txt"),
        help="plain-text corpus path (default: datasets/discord/input.txt)",
    )
    parser.add_argument(
        "--speaker-map",
        type=Path,
        help="speaker mapping path (default: speakers.json next to the corpus)",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        help="conversion statistics path (default: stats.json next to the corpus)",
    )
    parser.add_argument(
        "--session-gap-minutes",
        type=float,
        default=60.0,
        help="start a new conversation after this inactivity gap (default: 60)",
    )
    parser.add_argument(
        "--include-bots",
        action="store_true",
        help="include messages whose author is marked as a bot",
    )
    parser.add_argument(
        "--include-system-messages",
        action="store_true",
        help="include Discord notification message types in addition to messages/replies",
    )
    parser.add_argument(
        "--anonymize-speakers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "use USER_0001 aliases instead of Discord display names (default: enabled)"
        ),
    )
    parser.add_argument(
        "--reply-markers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="emit <REPLY_TO_USER> markers before replies (default: disabled)",
    )
    parser.add_argument(
        "--merge-consecutive-messages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="merge short consecutive messages from the same speaker (default: enabled)",
    )
    parser.add_argument(
        "--merge-message-gap-seconds",
        type=float,
        default=60.0,
        help="maximum gap between messages merged into one speaker block (default: 60)",
    )
    parser.add_argument(
        "--merge-message-max-characters",
        type=int,
        default=160,
        help="maximum size of each message eligible for merging (default: 160)",
    )
    parser.add_argument(
        "--merged-block-max-characters",
        type=int,
        default=400,
        help="maximum combined size of a merged speaker block (default: 400)",
    )
    parser.add_argument(
        "--media-placeholders",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="represent attachments, stickers, and embed-only messages (default: enabled)",
    )
    parser.add_argument(
        "--keep-urls",
        action="store_true",
        help="keep full URLs instead of replacing them with <URL>",
    )
    args = parser.parse_args()

    if args.session_gap_minutes <= 0:
        parser.error("--session-gap-minutes must be positive")
    if args.merge_message_gap_seconds <= 0:
        parser.error("--merge-message-gap-seconds must be positive")
    if args.merge_message_max_characters <= 0:
        parser.error("--merge-message-max-characters must be positive")
    if args.merged_block_max_characters <= 0:
        parser.error("--merged-block-max-characters must be positive")

    return args


def path_key(path: Path) -> str:
    return str(path.resolve())


def discover_json_files(inputs: list[Path], excluded_paths: set[Path]) -> list[Path]:
    discovered: dict[str, Path] = {}
    excluded = {path.resolve() for path in excluded_paths}

    for input_path in inputs:
        if not input_path.exists():
            raise FileNotFoundError(f"input path not found: {input_path}")

        if input_path.is_file():
            if input_path.suffix.lower() != ".json":
                raise ValueError(
                    f"input file must use the .json extension: {input_path}"
                )
            candidates = [input_path]
        elif input_path.is_dir():
            candidates = input_path.rglob("*.json")
        else:
            raise ValueError(
                f"input path is neither a file nor a directory: {input_path}"
            )

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in excluded:
                discovered[path_key(resolved)] = resolved

    files = [discovered[key] for key in sorted(discovered)]
    if not files:
        raise ValueError("no DiscordChatExporter JSON files found")
    return files


def require_object(value: Any, description: str, source_path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{source_path}: {description} must be a JSON object")
    return value


def require_list(value: Any, description: str, source_path: Path) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{source_path}: {description} must be a JSON array")
    return value


def require_string(value: Any, description: str, source_path: Path) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{source_path}: {description} must be a string")
    return value


def parse_timestamp(value: Any, description: str, source_path: Path) -> datetime:
    raw_timestamp = require_string(value, description, source_path)
    normalized_timestamp = raw_timestamp.replace("Z", "+00:00")
    try:
        timestamp = datetime.fromisoformat(normalized_timestamp)
    except ValueError as error:
        raise ValueError(
            f"{source_path}: invalid ISO 8601 {description}: {raw_timestamp!r}"
        ) from error

    if timestamp.tzinfo is None:
        raise ValueError(f"{source_path}: {description} must include a time zone")
    return timestamp.astimezone(timezone.utc)


def normalize_text(text: str, keep_urls: bool) -> str:
    normalized = unicodedata.normalize("NFC", text).replace("\r\n", "\n")
    normalized = normalized.replace("\r", "\n").replace("\x00", "\ufffd")
    if not keep_urls:
        normalized = URL_PATTERN.sub("<URL>", normalized)

    lines = [line.rstrip() for line in normalized.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def attachment_placeholder(attachment: dict[str, Any]) -> str:
    file_name = attachment.get("fileName")
    suffix = Path(file_name).suffix.lower() if isinstance(file_name, str) else ""
    if suffix in IMAGE_EXTENSIONS:
        return "<IMAGE>"
    if suffix in VIDEO_EXTENSIONS:
        return "<VIDEO>"
    if suffix in AUDIO_EXTENSIONS:
        return "<AUDIO>"
    return "<FILE>"


def message_content(
    message: dict[str, Any],
    source_path: Path,
    keep_urls: bool,
    media_placeholders: bool,
) -> str:
    raw_content = require_string(message.get("content"), "message.content", source_path)
    content = normalize_text(raw_content, keep_urls=keep_urls)
    if not media_placeholders:
        return content

    placeholders = []
    attachments = require_list(
        message.get("attachments", []),
        "message.attachments",
        source_path,
    )
    for attachment_value in attachments:
        attachment = require_object(
            attachment_value,
            "message attachment",
            source_path,
        )
        placeholders.append(attachment_placeholder(attachment))

    stickers = require_list(
        message.get("stickers", []), "message.stickers", source_path
    )
    placeholders.extend("<STICKER>" for _ in stickers)

    embeds = require_list(message.get("embeds", []), "message.embeds", source_path)
    if not content and embeds:
        placeholders.extend("<EMBED>" for _ in embeds)

    if not placeholders:
        return content
    if not content:
        return " ".join(placeholders)
    return f"{content}\n{' '.join(placeholders)}"


def optional_edited_timestamp(
    message: dict[str, Any], source_path: Path
) -> datetime | None:
    value = message.get("timestampEdited")
    if value is None:
        return None
    return parse_timestamp(value, "message.timestampEdited", source_path)


def optional_reference_message_id(
    message: dict[str, Any], source_path: Path
) -> str | None:
    reference_value = message.get("reference")
    if reference_value is None:
        return None
    reference = require_object(reference_value, "message.reference", source_path)
    message_id = reference.get("messageId")
    if message_id is None:
        return None
    return require_string(message_id, "message.reference.messageId", source_path)


def parse_message(
    value: Any,
    channel_id: str,
    source_path: Path,
    keep_urls: bool,
    media_placeholders: bool,
) -> DiscordMessage:
    message = require_object(value, "message", source_path)
    author = require_object(message.get("author"), "message.author", source_path)

    is_bot = author.get("isBot")
    if not isinstance(is_bot, bool):
        raise ValueError(f"{source_path}: message.author.isBot must be a boolean")

    author_name = require_string(author.get("name"), "message.author.name", source_path)
    nickname_value = author.get("nickname")
    if nickname_value is None:
        author_nickname = author_name
    else:
        author_nickname = require_string(
            nickname_value,
            "message.author.nickname",
            source_path,
        )

    return DiscordMessage(
        id=require_string(message.get("id"), "message.id", source_path),
        channel_id=channel_id,
        timestamp=parse_timestamp(
            message.get("timestamp"), "message.timestamp", source_path
        ),
        timestamp_edited=optional_edited_timestamp(message, source_path),
        message_type=require_string(message.get("type"), "message.type", source_path),
        author_id=require_string(author.get("id"), "message.author.id", source_path),
        author_name=author_name,
        author_nickname=author_nickname,
        is_bot=is_bot,
        content=message_content(
            message,
            source_path=source_path,
            keep_urls=keep_urls,
            media_placeholders=media_placeholders,
        ),
        reference_message_id=optional_reference_message_id(message, source_path),
        source_path=source_path,
    )


def load_export(
    path: Path,
    keep_urls: bool,
    media_placeholders: bool,
) -> list[DiscordMessage]:
    try:
        with path.open(encoding="utf-8") as file:
            payload = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{path}: invalid JSON at line {error.lineno}, column {error.colno}"
        ) from error

    root = require_object(payload, "export root", path)
    channel = require_object(root.get("channel"), "channel", path)
    channel_id = require_string(channel.get("id"), "channel.id", path)
    messages = require_list(root.get("messages"), "messages", path)
    return [
        parse_message(
            value,
            channel_id=channel_id,
            source_path=path,
            keep_urls=keep_urls,
            media_placeholders=media_placeholders,
        )
        for value in messages
    ]


def choose_duplicate(
    current: DiscordMessage,
    candidate: DiscordMessage,
) -> DiscordMessage:
    if current.duplicate_signature() == candidate.duplicate_signature():
        if current.timestamp_edited is None:
            return candidate if candidate.timestamp_edited is not None else current
        if candidate.timestamp_edited is None:
            return current
        return (
            candidate
            if candidate.timestamp_edited > current.timestamp_edited
            else current
        )

    same_message_identity = (
        current.channel_id == candidate.channel_id
        and current.timestamp == candidate.timestamp
        and current.author_id == candidate.author_id
    )
    if same_message_identity and current.timestamp_edited != candidate.timestamp_edited:
        if current.timestamp_edited is None:
            return candidate
        if candidate.timestamp_edited is None:
            return current
        return (
            candidate
            if candidate.timestamp_edited > current.timestamp_edited
            else current
        )

    raise ValueError(
        f"message {current.id} has conflicting versions in "
        f"{current.source_path} and {candidate.source_path}"
    )


def discord_id_key(value: str) -> tuple[int, int | str]:
    if value.isdigit():
        return (0, int(value))
    return (1, value)


def message_sort_key(message: DiscordMessage) -> tuple[datetime, tuple[int, int | str]]:
    return (message.timestamp, discord_id_key(message.id))


def build_conversations(
    messages: list[DiscordMessage],
    session_gap: timedelta,
) -> list[Conversation]:
    messages_by_channel: dict[str, list[DiscordMessage]] = defaultdict(list)
    for message in messages:
        messages_by_channel[message.channel_id].append(message)

    conversations = []
    for channel_id, channel_messages in messages_by_channel.items():
        channel_messages.sort(key=message_sort_key)
        current_messages: list[DiscordMessage] = []
        previous_timestamp: datetime | None = None

        for message in channel_messages:
            if (
                previous_timestamp is not None
                and message.timestamp - previous_timestamp >= session_gap
            ):
                conversations.append(
                    Conversation(
                        channel_id=channel_id, messages=tuple(current_messages)
                    )
                )
                current_messages = []

            current_messages.append(message)
            previous_timestamp = message.timestamp

        if current_messages:
            conversations.append(
                Conversation(channel_id=channel_id, messages=tuple(current_messages))
            )

    conversations.sort(
        key=lambda conversation: (
            conversation.started_at,
            discord_id_key(conversation.channel_id),
        )
    )
    return conversations


def most_common_text(values: Counter[str]) -> str:
    return min(values, key=lambda value: (-values[value], value))


def sanitize_speaker_label(display_name: str) -> str:
    normalized = unicodedata.normalize("NFC", display_name).strip()
    label_characters = []
    for character in normalized:
        if character.isspace():
            label_characters.append("_")
        elif character not in "<>" and not unicodedata.category(character).startswith(
            "C"
        ):
            label_characters.append(character)

    label = re.sub(r"_+", "_", "".join(label_characters)).strip("_")
    if not label:
        label = "USER"
    if label in RESERVED_SPEAKER_LABELS or label.startswith("REPLY_TO_"):
        label = f"{label}_SPEAKER"
    return label


def build_display_name_labels(
    author_ids: list[str],
    display_names: dict[str, str],
) -> dict[str, str]:
    labels: dict[str, str] = {}
    used_labels: set[str] = set()
    for author_id in author_ids:
        base_label = sanitize_speaker_label(display_names[author_id])
        label = base_label
        suffix = 2
        while label in used_labels:
            label = f"{base_label}_{suffix}"
            suffix += 1
        labels[author_id] = label
        used_labels.add(label)
    return labels


def build_speaker_data(
    messages: list[DiscordMessage],
    anonymize_speakers: bool,
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    names_by_author: dict[str, Counter[str]] = defaultdict(Counter)
    nicknames_by_author: dict[str, Counter[str]] = defaultdict(Counter)
    bots_by_author: dict[str, bool] = {}
    messages_by_author: Counter[str] = Counter()
    characters_by_author: Counter[str] = Counter()

    for message in messages:
        names_by_author[message.author_id][message.author_name] += 1
        nicknames_by_author[message.author_id][message.author_nickname] += 1
        bots_by_author[message.author_id] = message.is_bot
        messages_by_author[message.author_id] += 1
        characters_by_author[message.author_id] += len(message.content)

    author_ids = sorted(names_by_author, key=discord_id_key)
    display_names = {
        author_id: most_common_text(nicknames_by_author[author_id])
        for author_id in author_ids
    }
    if anonymize_speakers:
        aliases = {
            author_id: f"USER_{index:04d}"
            for index, author_id in enumerate(author_ids, start=1)
        }
    else:
        aliases = build_display_name_labels(author_ids, display_names)

    speaker_data = {}
    for author_id in author_ids:
        alias = aliases[author_id]
        speaker_data[alias] = {
            "author_id": author_id,
            "display_name": display_names[author_id],
            "username": most_common_text(names_by_author[author_id]),
            "is_bot": bots_by_author[author_id],
            "message_count": messages_by_author[author_id],
            "character_count": characters_by_author[author_id],
        }
    return aliases, speaker_data


def sanitize_discord_markup(content: str, aliases: dict[str, str]) -> str:
    def replace_user_mention(match: re.Match[str]) -> str:
        alias = aliases.get(match.group(1), "USER_UNKNOWN")
        return f"@{alias}"

    content = USER_MENTION_PATTERN.sub(replace_user_mention, content)
    content = ROLE_MENTION_PATTERN.sub("@ROLE", content)
    content = CHANNEL_MENTION_PATTERN.sub("<CHANNEL>", content)
    content = CUSTOM_EMOJI_PATTERN.sub(lambda match: f":{match.group(1)}:", content)
    return DISCORD_TIMESTAMP_PATTERN.sub("<TIMESTAMP>", content)


def reply_marker(
    message: DiscordMessage,
    messages_by_id: dict[str, DiscordMessage],
    aliases: dict[str, str],
) -> str | None:
    if message.message_type != "Reply":
        return None
    if message.reference_message_id is None:
        return "<REPLY>"

    referenced_message = messages_by_id.get(message.reference_message_id)
    if referenced_message is None:
        return "<REPLY>"
    referenced_alias = aliases.get(referenced_message.author_id)
    if referenced_alias is None:
        return "<REPLY>"
    return f"<REPLY_TO_{referenced_alias}>"


def build_message_blocks(
    messages: tuple[DiscordMessage, ...],
    merge_consecutive_messages: bool,
    merge_message_gap: timedelta,
    merge_message_max_characters: int,
    merged_block_max_characters: int,
) -> list[tuple[DiscordMessage, ...]]:
    blocks: list[list[DiscordMessage]] = []
    block_character_counts: list[int] = []

    for message in messages:
        if not blocks:
            blocks.append([message])
            block_character_counts.append(len(message.content))
            continue

        current_block = blocks[-1]
        previous_message = current_block[-1]
        combined_character_count = block_character_counts[-1] + 1 + len(message.content)
        should_merge = (
            merge_consecutive_messages
            and message.author_id == previous_message.author_id
            and message.timestamp - previous_message.timestamp <= merge_message_gap
            and len(previous_message.content) <= merge_message_max_characters
            and len(message.content) <= merge_message_max_characters
            and combined_character_count <= merged_block_max_characters
        )
        if should_merge:
            current_block.append(message)
            block_character_counts[-1] = combined_character_count
        else:
            blocks.append([message])
            block_character_counts.append(len(message.content))

    return [tuple(block) for block in blocks]


def render_corpus(
    output_path: Path,
    conversations: list[Conversation],
    aliases: dict[str, str],
    messages_by_id: dict[str, DiscordMessage],
    reply_markers: bool,
    merge_consecutive_messages: bool,
    merge_message_gap: timedelta,
    merge_message_max_characters: int,
    merged_block_max_characters: int,
) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.name}.tmp")
    character_count = 0
    speaker_block_count = 0

    with temporary_path.open("w", encoding="utf-8", newline="\n") as output:
        for conversation_index, conversation in enumerate(conversations):
            if conversation_index:
                output.write("\n\n")
                character_count += 2

            output.write(f"{CONVERSATION_START}\n")
            character_count += len(CONVERSATION_START) + 1
            message_blocks = build_message_blocks(
                conversation.messages,
                merge_consecutive_messages=merge_consecutive_messages,
                merge_message_gap=merge_message_gap,
                merge_message_max_characters=merge_message_max_characters,
                merged_block_max_characters=merged_block_max_characters,
            )
            speaker_block_count += len(message_blocks)
            for message_block in message_blocks:
                alias = aliases[message_block[0].author_id]
                rendered_contents = []
                for message in message_block:
                    content = sanitize_discord_markup(message.content, aliases)
                    marker = (
                        reply_marker(message, messages_by_id, aliases)
                        if reply_markers
                        else None
                    )
                    rendered_contents.append(
                        f"{marker}\n{content}" if marker else content
                    )

                rendered_content = "\n".join(rendered_contents)
                rendered_message = f"<{alias}>\n{rendered_content}\n"
                output.write(rendered_message)
                character_count += len(rendered_message)

            output.write(CONVERSATION_END)
            character_count += len(CONVERSATION_END)

    temporary_path.replace(output_path)
    return character_count, speaker_block_count


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_path = args.output
    speaker_map_path = (
        args.speaker_map
        if args.speaker_map is not None
        else output_path.with_name("speakers.json")
    )
    stats_path = (
        args.stats if args.stats is not None else output_path.with_name("stats.json")
    )
    input_paths = discover_json_files(
        args.inputs,
        excluded_paths={output_path, speaker_map_path, stats_path},
    )

    messages_by_id: dict[str, DiscordMessage] = {}
    raw_message_count = 0
    duplicate_count = 0
    for input_path in input_paths:
        file_messages = load_export(
            input_path,
            keep_urls=args.keep_urls,
            media_placeholders=args.media_placeholders,
        )
        raw_message_count += len(file_messages)
        for message in file_messages:
            current = messages_by_id.get(message.id)
            if current is None:
                messages_by_id[message.id] = message
            else:
                duplicate_count += 1
                messages_by_id[message.id] = choose_duplicate(current, message)

    unique_messages = list(messages_by_id.values())
    skipped_bots = sum(message.is_bot for message in unique_messages)
    if not args.include_bots:
        unique_messages = [message for message in unique_messages if not message.is_bot]

    skipped_system_messages = sum(
        message.message_type not in INCLUDED_MESSAGE_TYPES
        for message in unique_messages
    )
    if not args.include_system_messages:
        unique_messages = [
            message
            for message in unique_messages
            if message.message_type in INCLUDED_MESSAGE_TYPES
        ]

    skipped_empty_messages = sum(not message.content for message in unique_messages)
    included_messages = [message for message in unique_messages if message.content]
    if not included_messages:
        raise ValueError("no usable messages remain after filtering")

    aliases, speaker_data = build_speaker_data(
        included_messages,
        anonymize_speakers=args.anonymize_speakers,
    )
    conversations = build_conversations(
        included_messages,
        session_gap=timedelta(minutes=args.session_gap_minutes),
    )
    character_count, speaker_block_count = render_corpus(
        output_path,
        conversations=conversations,
        aliases=aliases,
        messages_by_id=messages_by_id,
        reply_markers=args.reply_markers,
        merge_consecutive_messages=args.merge_consecutive_messages,
        merge_message_gap=timedelta(seconds=args.merge_message_gap_seconds),
        merge_message_max_characters=args.merge_message_max_characters,
        merged_block_max_characters=args.merged_block_max_characters,
    )

    write_json(
        speaker_map_path,
        {
            "schema_version": 1,
            "description": "Private alias mapping for the generated Discord corpus.",
            "speakers": speaker_data,
        },
    )
    write_json(
        stats_path,
        {
            "schema_version": 1,
            "source_format": "DiscordChatExporter JSON",
            "configuration": {
                "session_gap_minutes": args.session_gap_minutes,
                "include_bots": args.include_bots,
                "include_system_messages": args.include_system_messages,
                "anonymize_speakers": args.anonymize_speakers,
                "reply_markers": args.reply_markers,
                "merge_consecutive_messages": args.merge_consecutive_messages,
                "merge_message_gap_seconds": args.merge_message_gap_seconds,
                "merge_message_max_characters": args.merge_message_max_characters,
                "merged_block_max_characters": args.merged_block_max_characters,
                "media_placeholders": args.media_placeholders,
                "keep_urls": args.keep_urls,
            },
            "counts": {
                "source_files": len(input_paths),
                "raw_messages": raw_message_count,
                "duplicate_messages": duplicate_count,
                "unique_messages": len(messages_by_id),
                "skipped_bot_messages": 0 if args.include_bots else skipped_bots,
                "skipped_system_messages": (
                    0 if args.include_system_messages else skipped_system_messages
                ),
                "skipped_empty_messages": skipped_empty_messages,
                "included_messages": len(included_messages),
                "speaker_blocks": speaker_block_count,
                "merged_messages": len(included_messages) - speaker_block_count,
                "speakers": len(aliases),
                "channels": len({message.channel_id for message in included_messages}),
                "conversations": len(conversations),
                "corpus_characters": character_count,
                "corpus_bytes": output_path.stat().st_size,
            },
            "speaker_counts": {
                alias: {
                    "messages": data["message_count"],
                    "characters": data["character_count"],
                }
                for alias, data in speaker_data.items()
            },
        },
    )

    print(f"Read {len(input_paths):,} JSON file(s)")
    print(f"Read {raw_message_count:,} raw message(s)")
    print(f"Removed {duplicate_count:,} duplicate message(s)")
    print(
        f"Wrote {len(included_messages):,} message(s) in {len(conversations):,} conversation(s)"
    )
    print(
        f"Speaker blocks: {speaker_block_count:,} "
        f"({len(included_messages) - speaker_block_count:,} message(s) merged)"
    )
    print(f"Corpus characters: {character_count:,}")
    print(f"Corpus: {output_path}")
    print(f"Private speaker map: {speaker_map_path}")
    print(f"Statistics: {stats_path}")


if __name__ == "__main__":
    main()
