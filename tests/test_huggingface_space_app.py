from __future__ import annotations

import importlib.util
import json
from pathlib import Path


APP_PATH = Path(__file__).resolve().parents[1] / "integrations" / "huggingface-space" / "app.py"


def load_space_app():
    spec = importlib.util.spec_from_file_location("avw_hf_space_app", APP_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vertex_report_validation_passes_public_payload() -> None:
    app = load_space_app()

    report, result = app.validate_payload(app.EXAMPLE_VERTEX_REPORT, "Vertex run report", "avwpirate person", 10)

    assert result["ok"] is True
    assert "PASS" in report
    assert result["summary"]["worker_pool_count"] == 1


def test_vertex_report_validation_flags_private_references() -> None:
    app = load_space_app()
    payload = json.loads(app.EXAMPLE_VERTEX_REPORT)
    private_uri = "gs:/" + "/private-bucket/runs/output.mp4"
    home_path = "/home" + "/ivan/run"
    payload["logs"] = {"last_message": f"wrote {private_uri} from {home_path}"}

    _, result = app.validate_payload(json.dumps(payload), "Vertex run report", "avwpirate person", 10)

    assert result["ok"] is False
    codes = {finding["code"] for finding in result["publication_findings"]}
    assert "gcs_uri" in codes
    assert "local_home_path" in codes
    assert "generated_video_reference" in codes


def test_dataset_manifest_validation_accepts_jsonl() -> None:
    app = load_space_app()

    _, result = app.validate_payload(app.EXAMPLE_DATASET_MANIFEST, "Dataset manifest", "avwpirate person", 10)

    assert result["ok"] is True
    assert result["summary"]["row_count"] == 10
    assert result["summary"]["captions_with_trigger"] == 10


def test_dataset_manifest_validation_requires_trigger() -> None:
    app = load_space_app()
    rows = [
        {
            "file_name": "001.png",
            "caption": "cinematic portrait without the token",
            "width": 1024,
            "height": 1024,
        }
    ]

    _, result = app.validate_payload(json.dumps(rows), "Dataset manifest", "avwpirate person", 1)

    assert result["ok"] is False
    assert any("trigger token" in error for error in result["errors"])
