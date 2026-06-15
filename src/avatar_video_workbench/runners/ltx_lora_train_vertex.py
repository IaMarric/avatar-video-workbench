from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse


def main() -> int:
    workspace = Path(os.environ.get("AVW_WORKSPACE", "/workspace/avatar-video-workbench-ltx-lora"))
    shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True, exist_ok=True)

    _ensure_runtime_dependencies()
    from google.cloud import storage

    client = storage.Client()
    dataset_dir = workspace / "dataset"
    output_dir = workspace / "outputs" / "ltx_lora"
    repo_dir = workspace / "LTX-2"
    model_path = workspace / "models" / "ltx-2.3.safetensors"
    text_encoder_dir = workspace / "models" / "text_encoder"
    config_path = workspace / "configs" / "ltx_lora_training.yaml"

    _download_directory(client, _env("AVW_LTX_DATASET_URI"), dataset_dir)
    _download_file(client, _env("AVW_LTX_MODEL_URI"), model_path)
    _download_directory(client, _env("AVW_LTX_TEXT_ENCODER_URI"), text_encoder_dir)
    _clone_ltx_repo(repo_dir)
    dataset_json = _find_dataset_json(dataset_dir)
    precomputed_dir = dataset_json.parent / ".precomputed"
    _write_training_config(config_path, model_path, text_encoder_dir, precomputed_dir, output_dir)
    _run_trainer(repo_dir, dataset_json, model_path, text_encoder_dir, precomputed_dir, config_path)
    _upload_directory(client, output_dir, _env("AVW_LTX_OUTPUT_URI"))
    print(
        "AVW_LTX_LORA_TRAIN_RESULT="
        + json.dumps({"output_uri": _env("AVW_LTX_OUTPUT_URI"), "output_dir": str(output_dir)}, sort_keys=True),
        flush=True,
    )
    return 0


def _ensure_runtime_dependencies() -> None:
    missing = []
    for module_name, package_name in {"google.cloud.storage": "google-cloud-storage", "yaml": "PyYAML"}.items():
        try:
            __import__(module_name)
        except Exception:
            missing.append(package_name)
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", *sorted(set(missing))], check=True)


def _clone_ltx_repo(repo_dir: Path) -> None:
    repo_url = os.environ.get("AVW_LTX_TRAINER_REPO", "https://github.com/Lightricks/LTX-2.git")
    ref = os.environ.get("AVW_LTX_TRAINER_REF", "main")
    subprocess.run(["git", "clone", "--depth", "1", "--branch", ref, repo_url, str(repo_dir)], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "uv"], check=True)
    subprocess.run(["uv", "sync"], cwd=repo_dir, check=True)


def _find_dataset_json(dataset_dir: Path) -> Path:
    dataset_json = dataset_dir / "ltx_trainer" / "dataset.json"
    if dataset_json.is_file():
        return dataset_json
    dataset_json = dataset_dir / "dataset.json"
    if dataset_json.is_file():
        return dataset_json
    raise RuntimeError(f"LTX trainer dataset.json not found under {dataset_dir}")


def _run_trainer(
    repo_dir: Path,
    dataset_json: Path,
    model_path: Path,
    text_encoder_dir: Path,
    precomputed_dir: Path,
    config_path: Path,
) -> None:
    resolution_bucket = os.environ.get("AVW_LTX_RESOLUTION_BUCKET", "512x512x1")
    trigger = _env("AVW_LTX_TRIGGER")
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "packages/ltx-trainer/scripts/process_dataset.py",
            str(dataset_json),
            "--resolution-buckets",
            resolution_bucket,
            "--model-path",
            str(model_path),
            "--text-encoder-path",
            str(text_encoder_dir),
            "--lora-trigger",
            trigger,
        ],
        cwd=repo_dir,
        check=True,
    )
    if not precomputed_dir.is_dir():
        raise RuntimeError(f"LTX preprocessing did not create {precomputed_dir}")
    subprocess.run(["uv", "run", "python", "packages/ltx-trainer/scripts/train.py", str(config_path)], cwd=repo_dir, check=True)


def _write_training_config(
    config_path: Path,
    model_path: Path,
    text_encoder_dir: Path,
    precomputed_dir: Path,
    output_dir: Path,
) -> None:
    import yaml

    steps = int(os.environ.get("AVW_LTX_TRAINING_STEPS", "1200"))
    validation_prompt = os.environ.get(
        "AVW_LTX_VALIDATION_PROMPT",
        f"{_env('AVW_LTX_TRIGGER')} walks through a cinematic scene with stable identity and natural movement.",
    )
    config = {
        "model": {
            "model_path": str(model_path),
            "text_encoder_path": str(text_encoder_dir),
            "training_mode": "lora",
            "load_checkpoint": None,
        },
        "lora": {
            "rank": 32,
            "alpha": 32,
            "dropout": 0.0,
            "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
        },
        "training_strategy": {
            "name": "text_to_video",
            "first_frame_conditioning_p": 0.75,
            "with_audio": False,
        },
        "optimization": {
            "learning_rate": 1e-4,
            "steps": steps,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "optimizer_type": "adamw",
            "scheduler_type": "linear",
            "scheduler_params": {},
            "enable_gradient_checkpointing": True,
        },
        "acceleration": {
            "mixed_precision_mode": "bf16",
            "quantization": None,
            "load_text_encoder_in_8bit": False,
            "offload_optimizer_during_validation": False,
        },
        "data": {"preprocessed_data_root": str(precomputed_dir), "num_dataloader_workers": 2},
        "validation": {
            "prompts": [validation_prompt],
            "negative_prompt": "worst quality, inconsistent identity, blurry, jittery, distorted face, text, watermark",
            "images": None,
            "video_dims": [512, 512, 17],
            "frame_rate": 8.0,
            "seed": 42,
            "inference_steps": 20,
            "interval": 250,
            "guidance_scale": 4.0,
            "stg_scale": 1.0,
            "stg_blocks": [29],
            "stg_mode": "stg_v",
            "generate_audio": False,
            "skip_initial_validation": True,
        },
        "checkpoints": {"interval": 250, "keep_last_n": 3, "precision": "bfloat16"},
        "flow_matching": {"timestep_sampling_mode": "shifted_logit_normal", "timestep_sampling_params": {}},
        "hub": {"push_to_hub": False, "hub_model_id": None},
        "wandb": {
            "enabled": False,
            "project": "avatar-video-workbench",
            "entity": None,
            "tags": ["ltx2", "lora", "avatar-video-workbench"],
            "log_validation_videos": False,
        },
        "seed": 42,
        "output_dir": str(output_dir),
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _download_file(client, uri: str, destination: Path) -> None:
    bucket, blob = _parse_gcs_uri(uri)
    destination.parent.mkdir(parents=True, exist_ok=True)
    client.bucket(bucket).blob(blob).download_to_filename(destination)
    print(f"Downloaded {uri} to {destination}", flush=True)


def _download_directory(client, uri: str, destination: Path) -> None:
    bucket_name, prefix = _parse_gcs_uri(uri.rstrip("/") + "/")
    bucket = client.bucket(bucket_name)
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    for blob in client.list_blobs(bucket, prefix=prefix):
        if blob.name.endswith("/"):
            continue
        relative = blob.name[len(prefix) :]
        if not relative:
            continue
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(path)
        count += 1
    if count == 0:
        raise RuntimeError(f"No files found at {uri}")
    print(f"Downloaded {count} files from {uri} to {destination}", flush=True)


def _upload_directory(client, source: Path, destination_uri: str) -> int:
    bucket_name, prefix = _parse_gcs_uri(destination_uri.rstrip("/") + "/")
    bucket = client.bucket(bucket_name)
    count = 0
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source).as_posix()
        bucket.blob(prefix + relative).upload_from_filename(path)
        count += 1
    print(f"Uploaded {count} files to {destination_uri}", flush=True)
    return count


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc:
        raise ValueError(f"Expected gs:// URI, got {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
