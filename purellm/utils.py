from pathlib import Path

import torch


def create_run_directory(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_number = (
        max(
            (
                int(path.name)
                for path in output_dir.iterdir()
                if path.is_dir() and path.name.isdigit()
            ),
            default=0,
        )
        + 1
    )

    while True:
        run_directory = output_dir / str(run_number)
        try:
            run_directory.mkdir()
        except FileExistsError:
            run_number += 1
        else:
            return run_directory


def _get_device_rng_state(device: torch.device) -> torch.Tensor | None:
    if device.type == "cuda":
        return torch.cuda.get_rng_state(device)
    if device.type == "mps":
        return torch.mps.get_rng_state()
    return None


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
