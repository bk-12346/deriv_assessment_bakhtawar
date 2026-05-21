"""Input and output helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.constants import OUTPUTS_DIR


def load_json(path: str | Path) -> Any:
    """Load JSON from disk."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: str | Path, data: Any) -> None:
    """Write deterministic, human-readable JSON."""
    output_path = Path(path)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, indent=2, sort_keys=True, ensure_ascii=False)
        file.write("\n")


def ensure_outputs_dir() -> Path:
    """Create and return the outputs directory."""
    outputs_path = Path(OUTPUTS_DIR)
    outputs_path.mkdir(parents=True, exist_ok=True)
    return outputs_path


def iso_timestamp() -> str:
    """Return a UTC ISO-8601 timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def require_dict(value: Any, name: str) -> dict[str, Any]:
    """Validate that a value is a dictionary."""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def require_list(value: Any, name: str) -> list[Any]:
    """Validate that a value is a list."""
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    return value
