from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from avatar_video_workbench.backend_metadata import export_backend_metadata
from avatar_video_workbench.config import write_yaml
from avatar_video_workbench.vertex_reports import build_vertex_run_report

ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _validate(instance: Any, schema: dict[str, Any], path: str = "$", root: dict[str, Any] | None = None) -> None:
    root = schema if root is None else root

    if "$ref" in schema:
        ref = schema["$ref"]
        assert ref.startswith("#/$defs/"), f"Unsupported ref at {path}: {ref}"
        return _validate(instance, root["$defs"][ref.removeprefix("#/$defs/")], path, root)

    if "anyOf" in schema:
        if not any(_matches_schema(instance, item, root) for item in schema["anyOf"]):
            raise AssertionError(f"{path}: did not match any allowed schema")
        return

    if "const" in schema:
        assert instance == schema["const"], f"{path}: expected {schema['const']!r}, got {instance!r}"
    if "enum" in schema:
        assert instance in schema["enum"], f"{path}: expected one of {schema['enum']!r}, got {instance!r}"

    expected_type = schema.get("type")
    if expected_type is not None:
        allowed = expected_type if isinstance(expected_type, list) else [expected_type]
        assert any(_matches_type(instance, item) for item in allowed), f"{path}: expected {expected_type}, got {type(instance).__name__}"

    if "not" in schema:
        assert not _matches_schema(instance, schema["not"], root), f"{path}: matched forbidden schema"

    if isinstance(instance, dict):
        required = schema.get("required", [])
        for key in required:
            assert key in instance, f"{path}: missing required key {key!r}"

        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for key, value in instance.items():
            child_path = f"{path}.{key}"
            if key in properties:
                _validate(value, properties[key], child_path, root)
            elif additional is False:
                raise AssertionError(f"{child_path}: unexpected key")
            elif isinstance(additional, dict):
                _validate(value, additional, child_path, root)

    if isinstance(instance, list) and "items" in schema:
        for index, value in enumerate(instance):
            _validate(value, schema["items"], f"{path}[{index}]", root)

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema:
            assert instance >= schema["minimum"], f"{path}: expected >= {schema['minimum']}"
        if "exclusiveMinimum" in schema:
            assert instance > schema["exclusiveMinimum"], f"{path}: expected > {schema['exclusiveMinimum']}"


def _matches_type(instance: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(instance, dict)
    if expected == "array":
        return isinstance(instance, list)
    if expected == "string":
        return isinstance(instance, str)
    if expected == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if expected == "number":
        return isinstance(instance, (int, float)) and not isinstance(instance, bool)
    if expected == "boolean":
        return isinstance(instance, bool)
    raise AssertionError(f"Unsupported JSON Schema type: {expected}")


def _matches_schema(instance: Any, schema: dict[str, Any], root: dict[str, Any]) -> bool:
    if "anyOf" in schema:
        return any(_matches_schema(instance, item, root) for item in schema["anyOf"])
    if "$ref" in schema:
        ref = schema["$ref"].removeprefix("#/$defs/")
        return _matches_schema(instance, root["$defs"][ref], root)
    if schema.get("type") and not _matches_type(instance, schema["type"]):
        return False
    if "pattern" in schema:
        return isinstance(instance, str) and re.search(schema["pattern"], instance) is not None
    try:
        _validate(instance, schema, root=root)
    except AssertionError:
        return False
    return True


def test_backend_metadata_public_example_validates_against_schema() -> None:
    _validate(
        _load_json("docs/examples/ltx-i2v-backend-metadata.json"),
        _load_json("schemas/backend-metadata.schema.json"),
    )


def test_vertex_run_report_public_example_validates_against_schema() -> None:
    _validate(
        _load_json("docs/examples/vertex-run-report.json"),
        _load_json("schemas/vertex-run-report.schema.json"),
    )


def test_backend_metadata_schema_requires_top_level_fields() -> None:
    schema = _load_json("schemas/backend-metadata.schema.json")
    example = _load_json("docs/examples/ltx-i2v-backend-metadata.json")

    for field in schema["required"]:
        incomplete = dict(example)
        incomplete.pop(field)
        try:
            _validate(incomplete, schema)
        except AssertionError as exc:
            assert f"missing required key '{field}'" in str(exc)
        else:
            raise AssertionError(f"Schema allowed backend metadata without {field!r}")


def test_vertex_run_report_schema_requires_top_level_fields() -> None:
    schema = _load_json("schemas/vertex-run-report.schema.json")
    example = _load_json("docs/examples/vertex-run-report.json")

    for field in schema["required"]:
        incomplete = dict(example)
        incomplete.pop(field)
        try:
            _validate(incomplete, schema)
        except AssertionError as exc:
            assert f"missing required key '{field}'" in str(exc)
        else:
            raise AssertionError(f"Schema allowed vertex run report without {field!r}")


def test_public_schemas_reject_private_uri_fields(tmp_path: Path) -> None:
    backend_schema = _load_json("schemas/backend-metadata.schema.json")
    backend = _load_json("docs/examples/ltx-i2v-backend-metadata.json")
    backend["backend"]["private_uri"] = "gs://" + "private-bucket/run"

    vertex_schema = _load_json("schemas/vertex-run-report.schema.json")
    vertex = _load_json("docs/examples/vertex-run-report.json")
    vertex["logs"] = {"last_message": "read " + "gs://" + "private-bucket/run/input.png"}

    for instance, schema in [(backend, backend_schema), (vertex, vertex_schema)]:
        try:
            _validate(instance, schema)
        except AssertionError:
            pass
        else:
            raise AssertionError("Schema allowed a private URI field")


def test_generated_reports_validate_against_public_schemas(tmp_path: Path) -> None:
    config = tmp_path / "ltx_i2v.yaml"
    write_yaml(
        config,
        {
            "model_id": "dg845/LTX-2.3-Diffusers",
            "prompt": "demoavatar person, handheld phone video",
            "negative_prompt": "distorted, text, watermark",
            "width": 512,
            "height": 512,
            "num_frames": 17,
            "fps": 8,
            "seed": 1234,
            "num_inference_steps": 6,
            "guidance_scale": 2.5,
            "lora_weights_uri": "gs://" + "private-bucket/runs/avatar-lora/output/checkpoint.safetensors",
            "lora_scale": 0.85,
        },
    )
    backend = export_backend_metadata(config)
    vertex = build_vertex_run_report(
        {
            "name": "projects/private-project/locations/us-central1/customJobs/1234567890",
            "displayName": "avatar-ltx-smoke",
            "state": "JOB_STATE_SUCCEEDED",
            "createTime": "2026-06-15T15:00:00Z",
            "startTime": "2026-06-15T15:01:00Z",
            "endTime": "2026-06-15T15:03:30Z",
            "updateTime": "2026-06-15T15:03:45Z",
            "jobSpec": {"workerPoolSpecs": []},
        }
    )

    _validate(backend, _load_json("schemas/backend-metadata.schema.json"))
    _validate(vertex, _load_json("schemas/vertex-run-report.schema.json"))
