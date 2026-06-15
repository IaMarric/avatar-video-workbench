from __future__ import annotations

from pathlib import Path

from avatar_video_workbench.cloud import LtxI2VSubmitOptions, LtxLoraTrainSubmitOptions, _build_ltx_job_yaml, _build_ltx_lora_train_job_yaml


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
        model_uri="gs://" + "bucket/models/ltx.safetensors",
        text_encoder_uri="gs://" + "bucket/models/gemma",
        region="us-central1",
        container_image="region-docker.pkg.dev/project/repo/image:latest",
        machine_type="a3-highgpu-1g",
        accelerator_type="NVIDIA_H100_80GB",
        accelerator_count=1,
        boot_disk_type="pd-ssd",
        boot_disk_size_gb=1000,
        staging_dir=Path("runs/vertex-staging"),
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
