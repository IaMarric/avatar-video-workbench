from __future__ import annotations

import json
from pathlib import Path

from avatar_video_workbench.backend_metadata import export_backend_metadata
from avatar_video_workbench.config import write_json, write_yaml


def test_export_backend_metadata_from_ltx_config_redacts_lora_uri(tmp_path: Path) -> None:
    config = tmp_path / "ltx_i2v.yaml"
    write_yaml(
        config,
        {
            "model_id": "dg845/LTX-2.3-Diffusers",
            "prompt": "demoavatar person, handheld phone video, subtle head turn",
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

    metadata = export_backend_metadata(config)

    assert metadata["backend"]["name"] == "ltx2_i2v"
    assert metadata["model"]["id"] == "dg845/LTX-2.3-Diffusers"
    assert metadata["video"] == {"width": 512, "height": 512, "frames": 17, "fps": 8, "duration_seconds": 2.125}
    assert metadata["sampling"] == {"seed": 1234, "steps": 6, "guidance_scale": 2.5}
    assert metadata["lora"]["enabled"] is True
    assert metadata["lora"]["scale"] == 0.85
    assert metadata["lora"]["artifact"]["reference"] == "private_external_artifact"
    serialized = json.dumps(metadata, sort_keys=True)
    assert "private-bucket" not in serialized
    assert "safetensors" not in serialized
    assert "gs://" not in serialized


def test_export_backend_metadata_from_runtime_manifest_omits_paths(tmp_path: Path) -> None:
    manifest = tmp_path / "avatar_manifest.json"
    write_json(
        manifest,
        {
            "created_utc": "2026-06-15T00:00:00+00:00",
            "config_uri": "gs://" + "private-bucket/runs/avatar/config/ltx_i2v.yaml",
            "input_image_uri": "gs://" + "private-bucket/runs/avatar/input/source.png",
            "output_uri": "gs://" + "private-bucket/runs/avatar/output",
            "prompt": "demoavatar person, walking through a quiet street",
            "negative_prompt": "distorted, text, watermark",
            "generation": {
                "pipeline": "ltx2_i2v",
                "model_id": "dg845/LTX-2.3-Diffusers",
                "output_video": "video/avatar.mp4",
                "width": 512,
                "height": 512,
                "num_frames": 17,
                "fps": 8,
                "seed": 777,
                "num_inference_steps": 4,
                "guidance_scale": 2.0,
                "generation_seconds": 18.25,
                "video_bytes": 1024,
                "lora_weights_path": "/workspace/avatar/lora/checkpoint.safetensors",
                "lora_scale": 1.0,
            },
        },
    )

    metadata = export_backend_metadata(manifest)

    assert metadata["backend"]["provider"] == "vertex-ai"
    assert metadata["runtime_metrics"] == {"generation_seconds": 18.25, "video_bytes": 1024}
    assert metadata["comfyui_comparison"]["seed"] == 777
    assert metadata["comfyui_comparison"]["lora_enabled"] is True
    serialized = json.dumps(metadata, sort_keys=True)
    assert "private-bucket" not in serialized
    assert "/workspace" not in serialized
    assert "avatar.mp4" not in serialized


def test_export_backend_metadata_from_submission_manifest_uses_sibling_config(tmp_path: Path) -> None:
    staging = tmp_path / "avatar-run"
    config_dir = staging / "config"
    config_dir.mkdir(parents=True)
    write_yaml(
        config_dir / "ltx_i2v.yaml",
        {
            "model_id": "dg845/LTX-2.3-Diffusers",
            "prompt": "demoavatar person, seated camera turn",
            "negative_prompt": "distorted",
            "width": 256,
            "height": 256,
            "num_frames": 17,
            "fps": 8,
            "seed": 42,
            "num_inference_steps": 4,
            "guidance_scale": 2.5,
            "lora_weights_uri": None,
        },
    )
    submission = staging / "submission-manifest.json"
    write_json(
        submission,
        {
            "run_id": "avatar-run",
            "gcs": {"run": "gs://" + "private-bucket/runs/avatar-run"},
            "vertex_job_yaml": str(staging / "vertex-custom-job.yaml"),
            "submitted": False,
        },
    )

    metadata = export_backend_metadata(submission)

    assert metadata["source"]["source_type"] == "ltx_i2v_submission_manifest"
    assert metadata["backend"]["provider"] == "vertex-ai"
    assert metadata["lora"]["enabled"] is False
