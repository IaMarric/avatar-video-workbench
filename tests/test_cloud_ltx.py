from __future__ import annotations

from pathlib import Path

from avatar_video_workbench.cloud import LtxI2VSubmitOptions, LtxLoraTrainSubmitOptions, _build_ltx_job_yaml, _build_ltx_lora_train_job_yaml
from avatar_video_workbench.config import write_json
from avatar_video_workbench.runners.ltx_i2v_vertex import _runtime_dependencies
from avatar_video_workbench.runners.ltx_lora_train_vertex import _write_normalized_dataset_json


def test_ltx_i2v_runner_installs_peft_for_lora_loading() -> None:
    assert _runtime_dependencies()["peft"] == "peft"


def test_ltx_vertex_job_spec_contains_real_runner_env() -> None:
    options = LtxI2VSubmitOptions(
        run_id="avatar-ltx-smoke",
        gcs_root="gs://" + "bucket/runs",
        input_image=Path("input.png"),
        prompt="demoavatar person walking",
        negative_prompt="distorted, text",
        region="asia-southeast1",
        container_image="region-docker.pkg.dev/project/repo/image:latest",
        machine_type="a3-highgpu-1g",
        accelerator_type="NVIDIA_H100_80GB",
        accelerator_count=1,
        boot_disk_type="pd-ssd",
        boot_disk_size_gb=1000,
        staging_dir=Path("runs/vertex-staging"),
        lora_weights_uri="gs://" + "bucket/runs/avatar-lora/output/checkpoint.safetensors",
        lora_scale=0.9,
    )

    job = _build_ltx_job_yaml(
        options=options,
        runner_uri="gs://" + "bucket/runs/avatar-ltx-smoke/code/ltx_i2v_vertex.py",
        config_uri="gs://" + "bucket/runs/avatar-ltx-smoke/config/ltx_i2v.yaml",
        input_uri="gs://" + "bucket/runs/avatar-ltx-smoke/input/input.png",
        output_uri="gs://" + "bucket/runs/avatar-ltx-smoke/output",
    )

    spec = job["workerPoolSpecs"][0]
    assert spec["machineSpec"]["acceleratorType"] == "NVIDIA_H100_80GB"
    assert spec["containerSpec"]["imageUri"] == "region-docker.pkg.dev/project/repo/image:latest"
    env = {item["name"]: item["value"] for item in spec["containerSpec"]["env"]}
    assert env["AVW_LTX_RUNNER_URI"].endswith("/ltx_i2v_vertex.py")
    assert env["AVW_LTX_OUTPUT_URI"].endswith("/output")
    assert env["AVW_LTX_LORA_URI"].endswith("checkpoint.safetensors")


def test_ltx_lora_train_job_spec_uses_official_trainer_env() -> None:
    options = LtxLoraTrainSubmitOptions(
        run_id="avatar-ltx-lora",
        gcs_root="gs://" + "bucket/runs",
        dataset_dir=Path("runs/pirate"),
        trigger="avwpirate person",
        model_uri="hf://Lightricks/LTX-2/ltx-2-19b-dev.safetensors",
        text_encoder_uri="hf://Lightricks/LTX-2?include=text_encoder/**&include=tokenizer/**",
        region="us-central1",
        container_image="region-docker.pkg.dev/project/repo/image:latest",
        machine_type="a3-highgpu-1g",
        accelerator_type="NVIDIA_H100_80GB",
        accelerator_count=1,
        boot_disk_type="pd-ssd",
        boot_disk_size_gb=1000,
        staging_dir=Path("runs/vertex-staging"),
        hf_token_secret="projects/demo-project/secrets/hf-token",
    )

    job = _build_ltx_lora_train_job_yaml(
        options=options,
        runner_uri="gs://" + "bucket/runs/avatar-ltx-lora/code/ltx_lora_train_vertex.py",
        dataset_uri="gs://" + "bucket/runs/avatar-ltx-lora/dataset",
        output_uri="gs://" + "bucket/runs/avatar-ltx-lora/output",
    )

    spec = job["workerPoolSpecs"][0]
    assert spec["machineSpec"]["acceleratorType"] == "NVIDIA_H100_80GB"
    env = {item["name"]: item["value"] for item in spec["containerSpec"]["env"]}
    assert env["AVW_LTX_TRAIN_RUNNER_URI"].endswith("ltx_lora_train_vertex.py")
    assert env["AVW_LTX_TRIGGER"] == "avwpirate person"
    assert env["AVW_LTX_RESOLUTION_BUCKET"] == "512x512x1"
    assert env["AVW_LTX_MODEL_URI"] == "hf://Lightricks/LTX-2/ltx-2-19b-dev.safetensors"
    assert env["AVW_LTX_TEXT_ENCODER_URI"] == "hf://Lightricks/LTX-2?include=text_encoder/**&include=tokenizer/**"
    assert env["AVW_HF_TOKEN_SECRET"] == "projects/demo-project/secrets/hf-token"
    assert "AVW_SECRET_PROJECT" not in env
    assert "/usr/local/nvidia/lib64" in env["LD_LIBRARY_PATH"]


def test_ltx_lora_runner_normalizes_legacy_dataset_paths(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "run"
    legacy_dir = dataset_dir / "ltx_trainer"
    image_dir = dataset_dir / "dataset" / "images"
    legacy_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    (image_dir / "001.png").write_bytes(b"png")
    legacy_json = legacy_dir / "dataset.json"
    write_json(legacy_json, [{"media_path": "../dataset/images/001.png", "caption": "avwpirate person"}])

    normalized = _write_normalized_dataset_json(legacy_json, dataset_dir)

    assert normalized == dataset_dir / "dataset.json"
    assert '"media_path": "dataset/images/001.png"' in normalized.read_text(encoding="utf-8")
