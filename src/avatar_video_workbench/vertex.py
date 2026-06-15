from __future__ import annotations

from pathlib import Path
from string import Template
from typing import Any

import yaml

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


def preflight_vertex_job(job_yaml_path: Path) -> dict[str, Any]:
    job_yaml_path = job_yaml_path.expanduser().resolve()
    if not job_yaml_path.is_file():
        raise WorkbenchError(f"Vertex job file not found: {job_yaml_path}")
    data = yaml.safe_load(job_yaml_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise WorkbenchError(f"Expected a YAML mapping in {job_yaml_path}")

    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    if not data.get("displayName"):
        errors.append({"code": "missing_display_name", "path": "displayName"})

    specs = data.get("jobSpec", {}).get("workerPoolSpecs") if isinstance(data.get("jobSpec"), dict) else None
    if not isinstance(specs, list) or not specs:
        errors.append({"code": "missing_worker_pool", "path": "jobSpec.workerPoolSpecs"})
    else:
        for index, spec in enumerate(specs):
            _validate_worker_pool(spec, f"jobSpec.workerPoolSpecs[{index}]", errors, warnings)

    for path, value in _iter_strings(data):
        if _has_placeholder(value):
            errors.append({"code": "unresolved_placeholder", "path": path, "value": value})

    return {
        "job_yaml": str(job_yaml_path),
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
    }


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WorkbenchError(f"{label} must be a mapping")
    return value


def _flatten_env(env: dict[str, Any]) -> dict[str, str]:
    return {f"env_{key}": str(value) for key, value in env.items()}


def _validate_worker_pool(
    spec: Any,
    path: str,
    errors: list[dict[str, str]],
    warnings: list[dict[str, str]],
) -> None:
    if not isinstance(spec, dict):
        errors.append({"code": "worker_pool_not_mapping", "path": path})
        return
    machine = spec.get("machineSpec")
    container = spec.get("containerSpec")
    if not isinstance(machine, dict):
        errors.append({"code": "missing_machine_spec", "path": f"{path}.machineSpec"})
    else:
        for key in ["machineType", "acceleratorType", "acceleratorCount"]:
            if machine.get(key) in (None, ""):
                errors.append({"code": "missing_machine_field", "path": f"{path}.machineSpec.{key}"})
    if not isinstance(container, dict):
        errors.append({"code": "missing_container_spec", "path": f"{path}.containerSpec"})
        return
    if not container.get("imageUri"):
        errors.append({"code": "missing_image_uri", "path": f"{path}.containerSpec.imageUri"})
    if not container.get("command") and not container.get("args"):
        warnings.append({"code": "missing_command_or_args", "path": f"{path}.containerSpec"})


def _iter_strings(value: Any, path: str = "$") -> list[tuple[str, str]]:
    if isinstance(value, str):
        return [(path, value)]
    if isinstance(value, dict):
        rows: list[tuple[str, str]] = []
        for key, child in value.items():
            rows.extend(_iter_strings(child, f"{path}.{key}"))
        return rows
    if isinstance(value, list):
        rows = []
        for index, child in enumerate(value):
            rows.extend(_iter_strings(child, f"{path}[{index}]"))
        return rows
    return []


def _has_placeholder(value: str) -> bool:
    markers = [
        "YOUR_",
        "REGION-docker.pkg.dev/PROJECT",
        "/PROJECT/",
        "PROJECT/REPOSITORY",
        "${",
    ]
    return any(marker in value for marker in markers)
