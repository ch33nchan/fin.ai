from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import torch
from rich.console import Console

console = Console()


def get_device() -> torch.device:
    forced = os.getenv("FIN_DEVICE")
    if forced:
        forced = forced.lower()
        if forced == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if forced == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if forced == "cpu":
            return torch.device("cpu")

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
