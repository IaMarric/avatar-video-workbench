from __future__ import annotations

import json
from pathlib import Path

from avatar_video_workbench.config import write_json
from avatar_video_workbench.vertex_reports import VertexRunReportOptions, build_vertex_run_report, create_vertex_run_report


def _job_payload() -> dict:
    return {
        "name": "projects/private-project/locations/us-central1/customJobs/1234567890",
        "displayName": "avatar-ltx-smoke",
        "state": "JOB_STATE_SUCCEEDED",
        "createTime": "2026-06-15T15:00:00Z",
        "startTime": "2026-06-15T15:01:00Z",
        "endTime": "2026-06-15T15:03:30Z",
        "updateTime": "2026-06-15T15:03:45Z",
        "jobSpec": {
            "workerPoolSpecs": [
                {
                    "machineSpec": {
                        "machineType": "a3-highgpu-1g",
                        "acceleratorType": "NVIDIA_H100_80GB",
                        "acceleratorCount": 1,
                    },
                    "diskSpec": {
                        "bootDiskType": "pd-ssd",
                        "bootDiskSizeGb": 1000,
                    },
                    "replicaCount": "1",
                }
            ]
        },
    }


def test_build_vertex_run_report_sanitizes_project_and_summarizes_hardware() -> None:
    report = build_vertex_run_report(_job_payload())

    assert report["vertex"]["custom_job_id"] == "1234567890"
    assert report["vertex"]["location"] == "us-central1"
    assert report["vertex"]["timing"]["duration_seconds"] == 150.0
    assert report["vertex"]["hardware"]["worker_pools"] == [
        {
            "index": 0,
            "machine_type": "a3-highgpu-1g",
            "accelerator": {"type": "NVIDIA_H100_80GB", "count": 1},
            "replica_count": 1,
            "disk": {"boot_disk_type": "pd-ssd", "boot_disk_size_gb": 1000},
        }
    ]
    assert "private-project" not in json.dumps(report, sort_keys=True)


def test_build_vertex_run_report_estimates_accelerator_hours_across_replicas() -> None:
    payload = _job_payload()
    payload["startTime"] = "2026-06-15T15:00:00Z"
    payload["endTime"] = "2026-06-15T16:30:00Z"
    payload["jobSpec"]["workerPoolSpecs"][0]["machineSpec"]["acceleratorCount"] = 2
    payload["jobSpec"]["workerPoolSpecs"][0]["replicaCount"] = 3

    report = build_vertex_run_report(payload)

    assert report["vertex"]["cost_estimate"] == {
        "elapsed_runtime_seconds": 5400.0,
        "machine_types": ["a3-highgpu-1g"],
        "accelerator_count": 6,
        "estimated_accelerator_hours": 9.0,
    }


def test_build_vertex_run_report_summarizes_logs_and_output_metadata() -> None:
    output_metadata = {
        "backend": {"name": "ltx2_i2v", "runtime": "diffusers"},
        "model": {"id": "dg845/LTX-2.3-Diffusers"},
        "video": {"width": 512, "height": 512, "frames": 17, "fps": 8},
        "sampling": {"seed": 1234, "steps": 4},
        "lora": {"enabled": True, "scale": 0.9},
        "runtime_metrics": {"generation_seconds": 18.2},
        "privacy": {"generated_media_omitted": True},
    }
    logs = [
        {
            "timestamp": "2026-06-15T15:01:10Z",
            "severity": "INFO",
            "textPayload": "Downloaded " + "gs://" + "private-bucket/run/input.png",
        },
        {
            "timestamp": "2026-06-15T15:02:10Z",
            "severity": "ERROR",
            "jsonPayload": {"message": "projects/private-project failed to read model asset"},
        },
    ]

    report = build_vertex_run_report(_job_payload(), output_metadata=output_metadata, log_entries=logs)

    assert report["output"]["model"]["id"] == "dg845/LTX-2.3-Diffusers"
    assert report["logs"]["entry_count"] == 2
    assert report["logs"]["severity_counts"] == {"ERROR": 1, "INFO": 1}
    assert report["logs"]["last_error"] == "projects/[project] failed to read model asset"
    serialized = json.dumps(report, sort_keys=True)
    assert "private-bucket" not in serialized
    assert "gs://" not in serialized
    assert "private-project" not in serialized


def test_create_vertex_run_report_writes_json_from_files(tmp_path: Path) -> None:
    job_json = tmp_path / "job.json"
    metadata_json = tmp_path / "metadata.json"
    logs_json = tmp_path / "logs.json"
    out = tmp_path / "report.json"
    write_json(job_json, _job_payload())
    write_json(metadata_json, {"model": {"id": "dg845/LTX-2.3-Diffusers"}})
    write_json(logs_json, [{"severity": "INFO", "textPayload": "complete"}])

    report = create_vertex_run_report(
        VertexRunReportOptions(
            job_json=job_json,
            output_metadata=metadata_json,
            logs_json=logs_json,
            out_path=out,
        )
    )

    assert report["logs"]["last_message"] == "complete"
    assert json.loads(out.read_text(encoding="utf-8"))["vertex"]["state"] == "JOB_STATE_SUCCEEDED"
