import argparse
from pathlib import Path
from urllib.request import Request, urlopen

CHUNK_SIZE = 8 * 1024 * 1024
TINY_STORIES_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
)
DATASETS = {
    "tiny-stories": {
        "train.txt": f"{TINY_STORIES_URL}/TinyStoriesV2-GPT4-train.txt?download=true",
        "validation.txt": (
            f"{TINY_STORIES_URL}/TinyStoriesV2-GPT4-valid.txt?download=true"
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a PureLLM dataset")
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument(
        "--destination",
        type=Path,
        help="output directory (default: datasets/<dataset>)",
    )
    return parser.parse_args()


def download_file(url: str, destination: Path) -> None:
    if destination.is_file():
        print(f"Already downloaded: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_name(f"{destination.name}.part")
    downloaded_bytes = partial_path.stat().st_size if partial_path.exists() else 0
    headers = {"User-Agent": "PureLLM dataset downloader"}
    if downloaded_bytes:
        headers["Range"] = f"bytes={downloaded_bytes}-"

    request = Request(url, headers=headers)
    with urlopen(request) as response:
        can_resume = downloaded_bytes > 0 and response.status == 206
        if not can_resume:
            downloaded_bytes = 0

        content_length = response.headers.get("Content-Length")
        total_bytes = downloaded_bytes + int(content_length) if content_length else None
        mode = "ab" if can_resume else "wb"

        with partial_path.open(mode) as output:
            while chunk := response.read(CHUNK_SIZE):
                output.write(chunk)
                downloaded_bytes += len(chunk)
                if total_bytes is None:
                    progress = f"{downloaded_bytes / 1_000_000:.1f} MB"
                else:
                    percentage = downloaded_bytes / total_bytes * 100
                    progress = (
                        f"{downloaded_bytes / 1_000_000:.1f}/"
                        f"{total_bytes / 1_000_000:.1f} MB ({percentage:.1f}%)"
                    )
                print(
                    f"\rDownloading {destination.name}: {progress}", end="", flush=True
                )

    print()
    partial_path.replace(destination)
    print(f"Downloaded: {destination}")


def main() -> None:
    args = parse_args()
    destination = args.destination or Path("datasets") / args.dataset.replace("-", "_")
    for filename, url in DATASETS[args.dataset].items():
        download_file(url, destination / filename)


if __name__ == "__main__":
    main()
