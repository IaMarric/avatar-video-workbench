from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import yaml

from .config import WorkbenchError, write_json, write_yaml
from .runners import ltx_i2v_vertex, ltx_lora_train_vertex


@dataclass(frozen=True)
class LtxI2VSubmitOptions:
    run_id: str
    gcs_root: str
    input_image: Path
    prompt: str
    negative_prompt: str
    region: str
    container_image: str
    machine_type: str
    accelerator_type: str
    accelerator_count: int
    boot_disk_type: str
    boot_disk_size_gb: int
    staging_dir: Path
    model_id: str = "dg845/LTX-2.3-Diffusers"
    width: int = 256
    height: int = 256
    num_frames: int = 17
    fps: int = 8
    num_inference_steps: int = 4
    guidance_scale: float = 2.5
    seed: int = 1234
    spot: bool = True
    submit: bool = True
    lora_weights_uri: str | None = None
    lora_scale: float = 1.0


@dataclass(frozen=True)
class LtxLoraTrainSubmitOptions:
    run_id: str
    gcs_root: str
    dataset_dir: Path
    trigger: str
    model_uri: str
    text_encoder_uri: str
    region: str
    container_image: str
    machine_type: str
    accelerator_type: str
    accelerator_count: int
    boot_disk_type: str
    boot_disk_size_gb: int
    staging_dir: Path
    resolution_bucket: str = "512x512x1"
    training_steps: int = 1200
    trainer_repo: str = "https://github.com/Lightricks/LTX-2.git"
    trainer_ref: str = "main"
    validation_prompt: str | None = None
    spot: bool = True
    submit: bool = True


def submit_ltx_i2v(options: LtxI2VSubmitOptions) -> dict:
    _require_command("gcloud")
    _require_command("gsutil")
    _require_file(options.input_image, "input image")
    _validate_gcs_uri(options.gcs_root)
    if options.width % 32 or options.height % 32:
        raise WorkbenchError("LTX width and height must be divisible by 32")
    if (options.num_frames - 1) % 8:
        raise WorkbenchError("LTX num_frames must be 8k+1")

    run_gcs = options.gcs_root.rstrip("/") + "/" + options.run_id
    staging = options.staging_dir.expanduser().resolve() / options.run_id
    config_path = staging / "config" / "ltx_i2v.yaml"
    runner_path = staging / "code" / "ltx_i2v_vertex.py"
    job_path = staging / "vertex-custom-job.yaml"
    manifest_path = staging / "submission-manifest.json"
    input_path = staging / "input" / options.input_image.name
    staging.mkdir(parents=True, exist_ok=True)
    input_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(options.input_image.expanduser().resolve(), input_path)
    runner_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(ltx_i2v_vertex.__file__).resolve(), runner_path)

    output_filename = f"{_slug(options.run_id)}.mp4"
    config = {
        "model_id": options.model_id,
        "input_image_name": input_path.name,
        "output_filename": output_filename,
        "contact_sheet_filename": f"{_slug(options.run_id)}_contact_sheet.jpg",
        "manifest_filename": f"{_slug(options.run_id)}_manifest.json",
        "seed": options.seed,
        "fps": options.fps,
        "width": options.width,
        "height": options.height,
        "num_frames": options.num_frames,
        "num_inference_steps": options.num_inference_steps,
        "guidance_scale": options.guidance_scale,
        "stg_scale": 1.0,
        "modality_scale": 1.0,
        "guidance_rescale": 0.0,
        "audio_guidance_scale": options.guidance_scale,
        "audio_stg_scale": 1.0,
        "audio_modality_scale": 1.0,
        "audio_guidance_rescale": 0.0,
        "spatio_temporal_guidance_blocks": [28],
        "use_cross_timestep": True,
        "torch_dtype": "bfloat16",
        "cpu_offload": "sequential",
        "enable_vae_tiling": True,
        "output_type": "np",
        "prompt": options.prompt,
        "negative_prompt": options.negative_prompt,
        "lora_weights_uri": options.lora_weights_uri,
        "lora_scale": options.lora_scale,
    }
    write_yaml(config_path, config)

    runner_uri = f"{run_gcs}/code/{runner_path.name}"
    config_uri = f"{run_gcs}/config/{config_path.name}"
    input_uri = f"{run_gcs}/input/{input_path.name}"
    output_uri = f"{run_gcs}/output"
    job = _build_ltx_job_yaml(
        options=options,
        runner_uri=runner_uri,
        config_uri=config_uri,
        input_uri=input_uri,
        output_uri=output_uri,
    )
    write_yaml(job_path, job)
    submission = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": options.run_id,
        "region": options.region,
        "staging_dir": str(staging),
        "gcs": {
            "run": run_gcs,
            "runner": runner_uri,
            "config": config_uri,
            "input_image": input_uri,
            "output": output_uri,
        },
        "vertex_job_yaml": str(job_path),
        "submitted": False,
    }

    _run(["gsutil", "-m", "cp", str(runner_path), runner_uri])
    _run(["gsutil", "-m", "cp", str(config_path), config_uri])
    _run(["gsutil", "-m", "cp", str(input_path), input_uri])

    if options.submit:
        result = _run(
            [
                "gcloud",
                "ai",
                "custom-jobs",
                "create",
                f"--region={options.region}",
                f"--display-name={options.run_id}",
                f"--config={job_path}",
                "--format=json",
            ],
            capture=True,
        )
        payload = json.loads(result)
        submission["submitted"] = True
        submission["vertex_job"] = payload.get("name")
        submission["vertex_state"] = payload.get("state")
    write_json(manifest_path, submission)
    return submission


def submit_ltx_lora_train(options: LtxLoraTrainSubmitOptions) -> dict:
    _require_command("gcloud")
    _require_command("gsutil")
    _validate_gcs_uri(options.gcs_root)
    _validate_gcs_uri(options.model_uri)
    _validate_gcs_uri(options.text_encoder_uri)
    if not options.dataset_dir.expanduser().is_dir():
        raise WorkbenchError(f"dataset dir not found: {options.dataset_dir}")
    if not (options.dataset_dir.expanduser() / "ltx_trainer" / "dataset.json").is_file() and not (
        options.dataset_dir.expanduser() / "dataset.json"
    ).is_file():
        raise WorkbenchError("dataset dir must contain ltx_trainer/dataset.json or dataset.json")

    run_gcs = options.gcs_root.rstrip("/") + "/" + options.run_id
    staging = options.staging_dir.expanduser().resolve() / options.run_id
    runner_path = staging / "code" / "ltx_lora_train_vertex.py"
    job_path = staging / "vertex-ltx-lora-train.yaml"
    manifest_path = staging / "submission-manifest.json"
    runner_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(ltx_lora_train_vertex.__file__).resolve(), runner_path)

    dataset_uri = f"{run_gcs}/dataset"
    runner_uri = f"{run_gcs}/code/{runner_path.name}"
    output_uri = f"{run_gcs}/output"
    _run(["gsutil", "-m", "rsync", "-r", str(options.dataset_dir.expanduser().resolve()), dataset_uri])
    _run(["gsutil", "-m", "cp", str(runner_path), runner_uri])

    job = _build_ltx_lora_train_job_yaml(options=options, runner_uri=runner_uri, dataset_uri=dataset_uri, output_uri=output_uri)
    write_yaml(job_path, job)
    submission = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": options.run_id,
        "region": options.region,
        "staging_dir": str(staging),
        "gcs": {
            "run": run_gcs,
            "runner": runner_uri,
            "dataset": dataset_uri,
            "output": output_uri,
        },
        "vertex_job_yaml": str(job_path),
        "submitted": False,
    }
    if options.submit:
        result = _run(
            [
                "gcloud",
                "ai",
                "custom-jobs",
                "create",
                f"--region={options.region}",
                f"--display-name={options.run_id}",
                f"--config={job_path}",
                "--format=json",
            ],
            capture=True,
        )
        payload = json.loads(result)
        submission["submitted"] = True
        submission["vertex_job"] = payload.get("name")
        submission["vertex_state"] = payload.get("state")
    write_json(manifest_path, submission)
    return submission


def _build_ltx_job_yaml(
    *,
    options: LtxI2VSubmitOptions,
    runner_uri: str,
    config_uri: str,
    input_uri: str,
    output_uri: str,
) -> dict:
    env = [
        {"name": "AVW_WORKSPACE", "value": f"/workspace/{options.run_id}"},
        {"name": "AVW_LTX_RUNNER_URI", "value": runner_uri},
        {"name": "AVW_LTX_CONFIG_URI", "value": config_uri},
        {"name": "AVW_LTX_INPUT_IMAGE_URI", "value": input_uri},
        {"name": "AVW_LTX_OUTPUT_URI", "value": output_uri},
        {"name": "HF_HOME", "value": "/workspace/.cache/huggingface"},
        {"name": "HF_XET_HIGH_PERFORMANCE", "value": "1"},
        {"name": "HF_HUB_DISABLE_PROGRESS_BARS", "value": "1"},
        {"name": "TOKENIZERS_PARALLELISM", "value": "false"},
        {"name": "PYTORCH_ALLOC_CONF", "value": "expandable_segments:True"},
        {"name": "NVIDIA_VISIBLE_DEVICES", "value": "all"},
        {"name": "NVIDIA_DRIVER_CAPABILITIES", "value": "compute,utility"},
        {"name": "CUDA_VISIBLE_DEVICES", "value": "0"},
        {
            "name": "LD_LIBRARY_PATH",
            "value": "/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib",
        },
    ]
    if options.lora_weights_uri:
        env.append({"name": "AVW_LTX_LORA_URI", "value": options.lora_weights_uri})
    return {
        "scheduling": {"strategy": "SPOT" if options.spot else "STANDARD"},
        "workerPoolSpecs": [
            {
                "machineSpec": {
                    "machineType": options.machine_type,
                    "acceleratorType": options.accelerator_type,
                    "acceleratorCount": options.accelerator_count,
                },
                "diskSpec": {
                    "bootDiskType": options.boot_disk_type,
                    "bootDiskSizeGb": options.boot_disk_size_gb,
                },
                "replicaCount": 1,
                "containerSpec": {
                    "imageUri": options.container_image,
                    "command": ["bash", "-lc"],
                    "args": [
                        """set -euo pipefail
python3 -m pip install --quiet --upgrade google-cloud-storage
python3 - <<'PY'
import os
from pathlib import Path
from urllib.parse import urlparse
from google.cloud import storage
uri = os.environ["AVW_LTX_RUNNER_URI"]
parsed = urlparse(uri)
dest = Path("/workspace/runner/ltx_i2v_vertex.py")
dest.parent.mkdir(parents=True, exist_ok=True)
storage.Client().bucket(parsed.netloc).blob(parsed.path.lstrip("/")).download_to_filename(dest)
print(f"Downloaded Avatar Video Workbench LTX runner {uri} to {dest}", flush=True)
PY
python3 /workspace/runner/ltx_i2v_vertex.py
"""
                    ],
                    "env": env,
                },
            }
        ],
    }


def _build_ltx_lora_train_job_yaml(
    *,
    options: LtxLoraTrainSubmitOptions,
    runner_uri: str,
    dataset_uri: str,
    output_uri: str,
) -> dict:
    env = [
        {"name": "AVW_WORKSPACE", "value": f"/workspace/{options.run_id}"},
        {"name": "AVW_LTX_TRAIN_RUNNER_URI", "value": runner_uri},
        {"name": "AVW_LTX_DATASET_URI", "value": dataset_uri},
        {"name": "AVW_LTX_OUTPUT_URI", "value": output_uri},
        {"name": "AVW_LTX_MODEL_URI", "value": options.model_uri},
        {"name": "AVW_LTX_TEXT_ENCODER_URI", "value": options.text_encoder_uri},
        {"name": "AVW_LTX_TRIGGER", "value": options.trigger},
        {"name": "AVW_LTX_RESOLUTION_BUCKET", "value": options.resolution_bucket},
        {"name": "AVW_LTX_TRAINING_STEPS", "value": str(options.training_steps)},
        {"name": "AVW_LTX_TRAINER_REPO", "value": options.trainer_repo},
        {"name": "AVW_LTX_TRAINER_REF", "value": options.trainer_ref},
        {"name": "HF_HOME", "value": "/workspace/.cache/huggingface"},
        {"name": "HF_XET_HIGH_PERFORMANCE", "value": "1"},
        {"name": "TOKENIZERS_PARALLELISM", "value": "false"},
        {"name": "PYTORCH_ALLOC_CONF", "value": "expandable_segments:True"},
        {"name": "NVIDIA_VISIBLE_DEVICES", "value": "all"},
        {"name": "NVIDIA_DRIVER_CAPABILITIES", "value": "compute,utility"},
        {"name": "CUDA_VISIBLE_DEVICES", "value": "0"},
    ]
    if options.validation_prompt:
        env.append({"name": "AVW_LTX_VALIDATION_PROMPT", "value": options.validation_prompt})
    return {
        "scheduling": {"strategy": "SPOT" if options.spot else "STANDARD"},
        "workerPoolSpecs": [
            {
                "machineSpec": {
                    "machineType": options.machine_type,
                    "acceleratorType": options.accelerator_type,
                    "acceleratorCount": options.accelerator_count,
                },
                "diskSpec": {
                    "bootDiskType": options.boot_disk_type,
                    "bootDiskSizeGb": options.boot_disk_size_gb,
                },
                "replicaCount": 1,
                "containerSpec": {
                    "imageUri": options.container_image,
                    "command": ["bash", "-lc"],
                    "args": [
                        """set -euo pipefail
python3 -m pip install --quiet --upgrade google-cloud-storage
python3 - <<'PY'
import os
from pathlib import Path
from urllib.parse import urlparse
from google.cloud import storage
uri = os.environ["AVW_LTX_TRAIN_RUNNER_URI"]
parsed = urlparse(uri)
dest = Path("/workspace/runner/ltx_lora_train_vertex.py")
dest.parent.mkdir(parents=True, exist_ok=True)
storage.Client().bucket(parsed.netloc).blob(parsed.path.lstrip("/")).download_to_filename(dest)
print(f"Downloaded Avatar Video Workbench LTX LoRA trainer {uri} to {dest}", flush=True)
PY
python3 /workspace/runner/ltx_lora_train_vertex.py
"""
                    ],
                    "env": env,
                },
            }
        ],
    }


def _run(cmd: list[str], *, capture: bool = False) -> str:
    completed = subprocess.run(cmd, text=True, capture_output=capture, check=False)
    if completed.returncode != 0:
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        raise WorkbenchError(f"Command failed ({completed.returncode}): {' '.join(cmd)}\n{stdout}{stderr}")
    return (completed.stdout or "").strip() if capture else ""


def _require_command(name: str) -> None:
    if not shutil.which(name):
        raise WorkbenchError(f"Required command not found: {name}")


def _require_file(path: Path, label: str) -> None:
    if not path.expanduser().is_file():
        raise WorkbenchError(f"{label} not found: {path}")


def _validate_gcs_uri(uri: str) -> None:
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc:
        raise WorkbenchError(f"Expected GCS URI, got: {uri}")


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value.lower()).strip("-") or "ltx-i2v"
