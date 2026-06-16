from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import WorkbenchError, load_mapping, write_json


GCS_URI = re.compile(r"\bgs://[a-z0-9._-]+/[^\s'\"\)]*", re.IGNORECASE)
PROJECT_RESOURCE = re.compile(r"\bprojects/[^/\s]+")
HOME_PATH = re.compile(r"/home/[A-Za-z0-9._-]+/")


@dataclass(frozen=True)
class VertexRunReportOptions:
    out_path: Path
    job_json: Path | None = None
    job_name: str | None = None
    region: str | None = None
    logs_json: Path | None = None
    output_metadata: Path | None = None


def create_vertex_run_report(options: VertexRunReportOptions) -> dict[str, Any]:
    job_payload = _load_job_payload(options)
    output_metadata = load_mapping(options.output_metadata) if options.output_metadata else None
    logs = _load_log_entries(options.logs_json) if options.logs_json else None
    report = build_vertex_run_report(job_payload, output_metadata=output_metadata, log_entries=logs)
    write_json(options.out_path.expanduser().resolve(), report)
    return report


def build_vertex_run_report(
    job_payload: dict[str, Any],
    *,
    output_metadata: dict[str, Any] | None = None,
    log_entries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    job_name = str(job_payload.get("name") or "")
    timing = _timing(job_payload)
    cost_estimate = _cost_estimate(timing, job_payload)
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "source": {
            "tool": "avatar-video-workbench",
            "source_type": "vertex_custom_job",
        },
        "vertex": {
            "custom_job_id": _custom_job_id(job_name),
            "location": _location(job_name),
            "display_name": _sanitize_text(job_payload.get("displayName")),
            "state": job_payload.get("state"),
            "timing": timing,
            "hardware": {
                "worker_pools": _worker_pool_summaries(job_payload),
            },
            "cost_estimate": cost_estimate,
        },
        "privacy": {
            "project_ids_omitted": True,
            "bucket_names_omitted": True,
            "service_accounts_omitted": True,
            "generated_media_omitted": True,
        },
    }
    if output_metadata:
        report["output"] = _output_summary(output_metadata)
    if log_entries is not None:
        report["logs"] = _log_summary(log_entries)
    return _drop_none(report)


def _load_job_payload(options: VertexRunReportOptions) -> dict[str, Any]:
    if options.job_json and options.job_name:
        raise WorkbenchError("Use either --job-json or --job-name, not both")
    if options.job_json:
        return load_mapping(options.job_json)
    if not options.job_name:
        raise WorkbenchError("Either --job-json or --job-name is required")
    if not options.region:
        raise WorkbenchError("--region is required with --job-name")
    completed = subprocess.run(
        [
            "gcloud",
            "ai",
            "custom-jobs",
            "describe",
            options.job_name,
            f"--region={options.region}",
            "--format=json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise WorkbenchError(completed.stderr.strip() or "gcloud custom job describe failed")
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise WorkbenchError("gcloud custom job describe did not return a JSON object")
    return payload


def _load_log_entries(path: Path) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    if text.lstrip().startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise WorkbenchError(f"Expected a JSON array in {path}")
        return [item for item in payload if isinstance(item, dict)]
    entries = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise WorkbenchError(f"Expected JSON object in {path}:{line_number}")
        entries.append(item)
    return entries


def _worker_pool_summaries(job_payload: dict[str, Any]) -> list[dict[str, Any]]:
    job_spec = job_payload.get("jobSpec")
    if not isinstance(job_spec, dict):
        return []
    specs = job_spec.get("workerPoolSpecs")
    if not isinstance(specs, list):
        return []

    pools = []
    for index, spec in enumerate(specs):
        if not isinstance(spec, dict):
            continue
        machine = spec.get("machineSpec") if isinstance(spec.get("machineSpec"), dict) else {}
        disk = spec.get("diskSpec") if isinstance(spec.get("diskSpec"), dict) else {}
        pools.append(
            _drop_none(
                {
                    "index": index,
                    "machine_type": machine.get("machineType"),
                    "accelerator": _drop_none(
                        {
                            "type": machine.get("acceleratorType"),
                            "count": _optional_int(machine.get("acceleratorCount")),
                        }
                    ),
                    "replica_count": _optional_int(spec.get("replicaCount")),
                    "disk": _drop_none(
                        {
                            "boot_disk_type": disk.get("bootDiskType"),
                            "boot_disk_size_gb": _optional_int(disk.get("bootDiskSizeGb")),
                        }
                    ),
                }
            )
        )
    return pools


def _timing(job_payload: dict[str, Any]) -> dict[str, Any]:
    start = _parse_time(job_payload.get("startTime"))
    end = _parse_time(job_payload.get("endTime"))
    timing: dict[str, Any] = {
        "created_at": job_payload.get("createTime"),
        "started_at": job_payload.get("startTime"),
        "ended_at": job_payload.get("endTime"),
        "updated_at": job_payload.get("updateTime"),
    }
    if start and end:
        timing["duration_seconds"] = round((end - start).total_seconds(), 3)
    return _drop_none(timing)


def _cost_estimate(timing: dict[str, Any], job_payload: dict[str, Any]) -> dict[str, Any]:
    duration_seconds = timing.get("duration_seconds")
    pools = _worker_pool_summaries(job_payload)
    machine_types = sorted(
        {str(pool["machine_type"]) for pool in pools if pool.get("machine_type")}
    )
    accelerator_count = sum(_pool_accelerator_count(pool) for pool in pools)
    estimate: dict[str, Any] = {
        "elapsed_runtime_seconds": duration_seconds,
        "machine_types": machine_types,
        "accelerator_count": accelerator_count or None,
    }
    if isinstance(duration_seconds, int | float) and accelerator_count:
        estimate["estimated_accelerator_hours"] = round(
            duration_seconds * accelerator_count / 3600, 6
        )
    return _drop_none(estimate)


def _pool_accelerator_count(pool: dict[str, Any]) -> int:
    accelerator = pool.get("accelerator") if isinstance(pool.get("accelerator"), dict) else {}
    count = accelerator.get("count")
    replicas = pool.get("replica_count")
    return (count if isinstance(count, int) else 0) * (replicas if isinstance(replicas, int) else 1)


def _output_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    allowed = ["backend", "model", "video", "sampling", "lora", "runtime_metrics", "privacy"]
    return {
        key: _sanitize_value(metadata[key])
        for key in allowed
        if key in metadata and metadata[key] is not None
    }


def _log_summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    severities = Counter(str(entry.get("severity") or "DEFAULT") for entry in entries)
    messages = [_sanitize_text(_log_message(entry)) for entry in entries]
    messages = [message for message in messages if message]
    error_messages = [
        _sanitize_text(_log_message(entry))
        for entry in entries
        if str(entry.get("severity") or "").upper() in {"ERROR", "CRITICAL", "ALERT", "EMERGENCY"}
    ]
    timestamps = [entry.get("timestamp") for entry in entries if entry.get("timestamp")]
    return _drop_none(
        {
            "entry_count": len(entries),
            "severity_counts": dict(sorted(severities.items())),
            "first_timestamp": min(timestamps) if timestamps else None,
            "last_timestamp": max(timestamps) if timestamps else None,
            "last_message": messages[-1] if messages else None,
            "last_error": error_messages[-1] if error_messages else None,
        }
    )


def _log_message(entry: dict[str, Any]) -> str | None:
    for key in ["textPayload", "message"]:
        value = entry.get(key)
        if isinstance(value, str):
            return value
    json_payload = entry.get("jsonPayload")
    if isinstance(json_payload, dict):
        for key in ["message", "msg", "event"]:
            value = json_payload.get(key)
            if isinstance(value, str):
                return value
    return None


def _custom_job_id(name: str) -> str | None:
    match = re.search(r"/customJobs/([^/]+)$", name)
    if match:
        return match.group(1)
    return name or None


def _location(name: str) -> str | None:
    match = re.search(r"/locations/([^/]+)/", name)
    return match.group(1) if match else None


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, str):
        return _sanitize_text(value)
    return value


def _sanitize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    text = GCS_URI.sub("[gcs-uri]", text)
    text = PROJECT_RESOURCE.sub("projects/[project]", text)
    text = HOME_PATH.sub("/home/[user]/", text)
    return text


def _drop_none(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if item is None:
                continue
            normalized = _drop_none(item)
            if normalized in ({}, []):
                continue
            cleaned[key] = normalized
        return cleaned
    if isinstance(value, list):
        return [
            normalized
            for item in value
            if item is not None
            for normalized in [_drop_none(item)]
            if normalized not in ({}, [])
        ]
    return value
