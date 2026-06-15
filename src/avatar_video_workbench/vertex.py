from __future__ import annotations

from pathlib import Path
from string import Template
from typing import Any

from .config import WorkbenchError, load_mapping, require_keys


def render_vertex_job(config_path: Path, template_path: Path) -> str:
    config = load_mapping(config_path)
    require_keys(config, ["run", "vertex", "container"], label=str(config_path))

    run = _mapping(config["run"], "run")
    vertex = _mapping(config["vertex"], "vertex")
    container = _mapping(config["container"], "container")
    env = _flatten_env(config.get("env") or {})

    values: dict[str, str] = {}
    for section in [run, vertex, container, env]:
        for key, value in section.items():
            values[key] = str(value)

    required = [
        "display_name",
        "machine_type",
        "accelerator_type",
        "accelerator_count",
        "boot_disk_size_gb",
        "image_uri",
        "command",
    ]
    missing = [key for key in required if not values.get(key)]
    if missing:
        raise WorkbenchError(f"Vertex render config missing: {', '.join(missing)}")

    template = Template(template_path.read_text(encoding="utf-8"))
    try:
        return template.substitute(values)
    except KeyError as exc:
        raise WorkbenchError(f"Template variable not provided: {exc.args[0]}") from exc


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WorkbenchError(f"{label} must be a mapping")
    return value


def _flatten_env(env: dict[str, Any]) -> dict[str, str]:
    return {f"env_{key}": str(value) for key, value in env.items()}

