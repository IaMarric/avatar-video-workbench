from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - Spaces installs PyYAML.
    yaml = None


REPO_URL = "https://github.com/IaMarric/avatar-video-workbench"
RELEASE_URL = "https://github.com/IaMarric/avatar-video-workbench/releases/tag/v0.1.1"

SENSITIVE_PATTERNS = {
    "gcs_uri": re.compile(r"\bgs://[a-z0-9._-]+/[^\s'\"\)]*", re.IGNORECASE),
    "vertex_project_resource": re.compile(r"\bprojects/[^/\s]+"),
    "local_home_path": re.compile(r"/home/[A-Za-z0-9._-]+/"),
    "service_account": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.iam\.gserviceaccount\.com\b"),
    "huggingface_token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
}

MEDIA_PATTERNS = {
    "generated_video_reference": re.compile(r"\.(mp4|mov|webm|mkv)\b", re.IGNORECASE),
    "generated_image_reference": re.compile(r"\.(png|jpg|jpeg|webp)\b", re.IGNORECASE),
    "model_artifact_reference": re.compile(r"\.(safetensors|ckpt|pt|pth|onnx)\b", re.IGNORECASE),
}

NSFW_PATTERN = re.compile(r"\b(nsfw|nude|nudity|explicit|porn|sexual)\b", re.IGNORECASE)


EXAMPLE_VERTEX_REPORT = json.dumps(
    {
        "schema_version": "1.0",
        "source": {"tool": "avatar-video-workbench", "source_type": "vertex_custom_job"},
        "vertex": {
            "custom_job_id": "1234567890",
            "location": "us-central1",
            "display_name": "ltx-i2v-public-example",
            "state": "JOB_STATE_SUCCEEDED",
            "timing": {
                "created_at": "2026-06-15T10:00:00Z",
                "started_at": "2026-06-15T10:01:00Z",
                "ended_at": "2026-06-15T10:12:30Z",
                "duration_seconds": 690,
            },
            "hardware": {
                "worker_pools": [
                    {
                        "index": 0,
                        "machine_type": "a3-highgpu-1g",
                        "accelerator": {"type": "NVIDIA_H100_80GB", "count": 1},
                        "replica_count": 1,
                    }
                ]
            },
        },
        "output": {
            "backend": "ltx-2.3",
            "video": {"width": 512, "height": 512, "frames": 121, "fps": 24},
            "sampling": {"seed": 42, "steps": 30, "guidance_scale": 4.0},
            "lora": {"enabled": True, "scale": 1.0},
        },
        "privacy": {
            "project_ids_omitted": True,
            "bucket_names_omitted": True,
            "service_accounts_omitted": True,
            "generated_media_omitted": True,
        },
    },
    indent=2,
)

EXAMPLE_DATASET_MANIFEST = "\n".join(
    json.dumps(
        {
            "file_name": f"{index:03d}.png",
            "caption_file": f"{index:03d}.txt",
            "caption": f"avwpirate person, cinematic pirate portrait, safe public demo variant {index}",
            "trigger_present": True,
            "width": 1024,
            "height": 1024,
            "aspect_ratio": 1.0,
            "sha256": f"{index:064x}"[-64:],
        },
        sort_keys=True,
    )
    for index in range(1, 11)
)


def validate_payload(raw_payload: str, mode: str, trigger: str, min_rows: int = 10) -> tuple[str, dict[str, Any]]:
    payload, parse_messages = parse_payload(raw_payload)
    findings = scan_public_text(raw_payload)
    warnings = list(parse_messages)
    errors: list[str] = []

    if payload is None:
        errors.append("Input is not valid JSON, JSONL, or YAML.")
        result = _result(mode, errors, warnings, findings, {})
        return format_report(result), result

    if mode == "Vertex run report":
        summary, mode_errors, mode_warnings = validate_vertex_report(payload)
    elif mode == "Dataset manifest":
        summary, mode_errors, mode_warnings = validate_dataset_manifest(payload, trigger=trigger, min_rows=min_rows)
    else:
        summary, mode_errors, mode_warnings = {}, [f"Unknown mode: {mode}"], []

    errors.extend(mode_errors)
    warnings.extend(mode_warnings)
    if NSFW_PATTERN.search(raw_payload):
        errors.append("NSFW-related wording detected. Keep public examples family-safe.")

    result = _result(mode, errors, warnings, findings, summary)
    return format_report(result), result


def parse_payload(raw_payload: str) -> tuple[Any | None, list[str]]:
    text = raw_payload.strip()
    if not text:
        return None, ["No input provided."]

    try:
        return json.loads(text), []
    except json.JSONDecodeError:
        pass

    jsonl_rows = []
    jsonl_errors = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            jsonl_errors.append(f"Line {line_number}: {exc.msg}")
            break
        jsonl_rows.append(item)
    if jsonl_rows and not jsonl_errors:
        return jsonl_rows, []

    if yaml is not None:
        try:
            return yaml.safe_load(text), []
        except Exception:
            pass
    return None, jsonl_errors[:3]


def scan_public_text(text: str) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for code, pattern in SENSITIVE_PATTERNS.items():
        for match in pattern.finditer(text):
            findings.append({"severity": "error", "code": code, "sample": _sample(match.group(0))})
    for code, pattern in MEDIA_PATTERNS.items():
        for match in pattern.finditer(text):
            findings.append({"severity": "warning", "code": code, "sample": _sample(match.group(0))})
    return findings


def validate_vertex_report(payload: Any) -> tuple[dict[str, Any], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(payload, dict):
        return {}, ["Vertex run report must be a JSON/YAML object."], warnings

    source = _mapping(payload.get("source"))
    vertex = _mapping(payload.get("vertex"))
    privacy = _mapping(payload.get("privacy"))
    hardware = _mapping(vertex.get("hardware"))
    worker_pools = hardware.get("worker_pools") if isinstance(hardware.get("worker_pools"), list) else []
    output = _mapping(payload.get("output"))
    timing = _mapping(vertex.get("timing"))

    if payload.get("schema_version") != "1.0":
        warnings.append("Expected schema_version 1.0.")
    if source.get("tool") != "avatar-video-workbench":
        errors.append("source.tool must be avatar-video-workbench.")
    if source.get("source_type") != "vertex_custom_job":
        warnings.append("source.source_type should be vertex_custom_job.")
    if not vertex:
        errors.append("vertex section is required.")
    if not vertex.get("custom_job_id"):
        errors.append("vertex.custom_job_id is required and should not include the full project resource.")
    if not vertex.get("location"):
        warnings.append("vertex.location is missing.")

    required_privacy = [
        "project_ids_omitted",
        "bucket_names_omitted",
        "service_accounts_omitted",
        "generated_media_omitted",
    ]
    for key in required_privacy:
        if privacy.get(key) is not True:
            errors.append(f"privacy.{key} must be true for public sharing.")

    accelerator_counts = Counter()
    for pool in worker_pools:
        if not isinstance(pool, dict):
            continue
        accelerator = pool.get("accelerator") if isinstance(pool.get("accelerator"), dict) else {}
        if accelerator.get("type"):
            accelerator_counts[str(accelerator["type"])] += int(accelerator.get("count") or 0)

    summary = {
        "state": vertex.get("state"),
        "location": vertex.get("location"),
        "duration_seconds": timing.get("duration_seconds"),
        "worker_pool_count": len(worker_pools),
        "accelerators": dict(accelerator_counts),
        "output_sections": sorted(output.keys()),
    }
    return summary, errors, warnings


def validate_dataset_manifest(payload: Any, *, trigger: str, min_rows: int) -> tuple[dict[str, Any], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    rows = payload if isinstance(payload, list) else None
    if rows is None:
        return {}, ["Dataset manifest must be a JSON array or JSONL rows."], warnings

    if len(rows) < min_rows:
        warnings.append(f"Expected at least {min_rows} rows; found {len(rows)}.")

    file_names: list[str] = []
    captions_with_trigger = 0
    sizes = Counter()
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            errors.append(f"Row {index} must be an object.")
            continue

        file_name = row.get("file_name") or row.get("media_path")
        caption = str(row.get("caption") or "")
        if not file_name:
            errors.append(f"Row {index} is missing file_name or media_path.")
        else:
            file_names.append(str(file_name))
            if str(file_name).startswith(("/", "gs://")):
                errors.append(f"Row {index} uses a non-public file reference.")
        if not caption.strip():
            errors.append(f"Row {index} is missing caption.")
        if trigger and trigger in caption:
            captions_with_trigger += 1
        elif row.get("trigger_present") is True:
            captions_with_trigger += 1
        else:
            errors.append(f"Row {index} caption does not include the trigger token.")

        width = _optional_positive_int(row.get("width"))
        height = _optional_positive_int(row.get("height"))
        if width and height:
            sizes[f"{width}x{height}"] += 1
        elif "width" in row or "height" in row:
            warnings.append(f"Row {index} has incomplete or invalid dimensions.")

        sha256 = row.get("sha256")
        if sha256 is not None and not re.fullmatch(r"[a-fA-F0-9]{64}", str(sha256)):
            warnings.append(f"Row {index} sha256 is not a 64-character hex digest.")

    duplicate_files = sorted(name for name, count in Counter(file_names).items() if count > 1)
    if duplicate_files:
        errors.append(f"Duplicate file references: {', '.join(duplicate_files[:5])}.")

    summary = {
        "row_count": len(rows),
        "captions_with_trigger": captions_with_trigger,
        "unique_files": len(set(file_names)),
        "sizes": dict(sizes.most_common()),
    }
    return summary, errors, warnings


def format_report(result: dict[str, Any]) -> str:
    status = "PASS" if result["ok"] else "NEEDS ATTENTION"
    lines = [
        f"## {status}",
        "",
        f"Mode: `{result['mode']}`",
        f"Repository: [{REPO_URL}]({REPO_URL})",
        f"Release: [{RELEASE_URL}]({RELEASE_URL})",
        "",
    ]
    if result["summary"]:
        lines.extend(["### Summary", ""])
        for key, value in result["summary"].items():
            lines.append(f"- `{key}`: `{json.dumps(value, sort_keys=True)}`")
        lines.append("")
    if result["errors"]:
        lines.extend(["### Errors", ""])
        lines.extend(f"- {item}" for item in result["errors"])
        lines.append("")
    if result["warnings"]:
        lines.extend(["### Warnings", ""])
        lines.extend(f"- {item}" for item in result["warnings"])
        lines.append("")
    if result["publication_findings"]:
        lines.extend(["### Publication Findings", ""])
        for finding in result["publication_findings"]:
            lines.append(f"- `{finding['severity']}` `{finding['code']}`: `{finding['sample']}`")
        lines.append("")
    if result["ok"]:
        lines.append("This payload matches the selected public sharing checks.")
    return "\n".join(lines)


def _result(
    mode: str,
    errors: list[str],
    warnings: list[str],
    publication_findings: list[dict[str, str]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    has_error_findings = any(finding["severity"] == "error" for finding in publication_findings)
    return {
        "ok": not errors and not has_error_findings,
        "mode": mode,
        "summary": summary,
        "errors": errors,
        "warnings": warnings,
        "publication_findings": publication_findings,
    }


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _optional_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _sample(value: str) -> str:
    if len(value) <= 80:
        return value
    return value[:37] + "..." + value[-40:]


def build_app() -> Any:
    import gradio as gr

    with gr.Blocks(title="Avatar Video Workbench Report Checker") as demo:
        gr.Markdown("# Avatar Video Workbench Report Checker")
        with gr.Row():
            mode = gr.Radio(["Vertex run report", "Dataset manifest"], value="Vertex run report", label="Payload")
            trigger = gr.Textbox(value="avwpirate person", label="Trigger token")
            min_rows = gr.Number(value=10, precision=0, minimum=1, label="Minimum dataset rows")
        raw_payload = gr.Code(value=EXAMPLE_VERTEX_REPORT, language="json", label="JSON, JSONL, or YAML")
        with gr.Row():
            load_vertex = gr.Button("Load Vertex Example")
            load_dataset = gr.Button("Load Dataset Example")
            validate = gr.Button("Validate", variant="primary")
        report = gr.Markdown(label="Report")
        raw_result = gr.JSON(label="Structured Result")

        mode.change(
            lambda selected: EXAMPLE_DATASET_MANIFEST if selected == "Dataset manifest" else EXAMPLE_VERTEX_REPORT,
            inputs=mode,
            outputs=raw_payload,
        )
        load_vertex.click(lambda: ("Vertex run report", EXAMPLE_VERTEX_REPORT), outputs=[mode, raw_payload])
        load_dataset.click(lambda: ("Dataset manifest", EXAMPLE_DATASET_MANIFEST), outputs=[mode, raw_payload])
        validate.click(validate_payload, inputs=[raw_payload, mode, trigger, min_rows], outputs=[report, raw_result])
        demo.load(validate_payload, inputs=[raw_payload, mode, trigger, min_rows], outputs=[report, raw_result])
    return demo


if __name__ == "__main__":
    build_app().launch()
