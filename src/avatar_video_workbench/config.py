from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


class WorkbenchError(RuntimeError):
    """Raised for user-fixable workbench errors."""


def load_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise WorkbenchError(f"Config file not found: {path}")
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise WorkbenchError(f"Expected a mapping in {path}")
    return data


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=False), encoding="utf-8")


def require_keys(data: dict[str, Any], keys: list[str], *, label: str) -> None:
    missing = [key for key in keys if key not in data or data[key] in (None, "")]
    if missing:
        raise WorkbenchError(f"{label} missing required keys: {', '.join(missing)}")

