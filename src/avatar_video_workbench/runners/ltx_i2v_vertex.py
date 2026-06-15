from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


def main() -> int:
    config_uri = _env("AVW_LTX_CONFIG_URI")
    input_image_uri = _env("AVW_LTX_INPUT_IMAGE_URI")
    output_uri = _env("AVW_LTX_OUTPUT_URI")
    workspace = Path(os.environ.get("AVW_WORKSPACE", "/workspace/avatar-video-workbench-ltx"))

    config_path = workspace / "config" / "ltx_i2v.yaml"
    input_path = workspace / "input" / Path(urlparse(input_image_uri).path).name
    output_dir = workspace / "output"
    shutil.rmtree(workspace, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    _ensure_runtime_dependencies()
    from google.cloud import storage

    client = storage.Client()
    _download_file(client, config_uri, config_path)
    _download_file(client, input_image_uri, input_path)

    config = _read_yaml(config_path)
    shutil.copy2(config_path, output_dir / "ltx_i2v.yaml")
    shutil.copy2(input_path, output_dir / input_path.name)

    _log_gpu_status()
    generation = _run_ltx_i2v(config, input_path, output_dir)
    manifest_path = _write_manifest(output_dir, config, config_uri, input_image_uri, output_uri, generation)
    _make_video_contact_sheet(output_dir / generation["output_video"], output_dir / "reports" / config["contact_sheet_filename"])
    _upload_directory(client, output_dir, output_uri)
    print("AVW_LTX_VERTEX_RESULT=" + json.dumps({"manifest": str(manifest_path), "output_uri": output_uri}, sort_keys=True), flush=True)
    return 0


def _ensure_runtime_dependencies() -> None:
    required_imports = {
        "google.cloud.storage": "google-cloud-storage",
        "yaml": "PyYAML",
        "PIL": "Pillow",
        "torch": "torch",
        "transformers": "transformers",
        "accelerate": "accelerate",
        "sentencepiece": "sentencepiece",
        "google.protobuf": "protobuf",
        "av": "av",
    }
    missing = []
    for module_name, package_name in required_imports.items():
        try:
            __import__(module_name)
        except Exception:
            missing.append(package_name)

    try:
        from diffusers import LTX2ImageToVideoPipeline  # noqa: F401
        from diffusers.pipelines.ltx2.export_utils import encode_video  # noqa: F401
    except Exception:
        missing.append("git+https://github.com/huggingface/diffusers.git")

    if missing:
        command = [sys.executable, "-m", "pip", "install", "--upgrade", *sorted(set(missing))]
        print("Installing missing runtime packages: " + " ".join(command), flush=True)
        subprocess.run(command, check=True)


def _run_ltx_i2v(config: dict, input_image_path: Path, output_dir: Path) -> dict:
    import torch
    from diffusers import LTX2ImageToVideoPipeline
    from diffusers.pipelines.ltx2.export_utils import encode_video
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for LTX I2V generation")

    dtype = _torch_dtype(str(config.get("torch_dtype", "bfloat16")))
    model_id = str(config["model_id"])
    print("Loading LTX pipeline " + json.dumps({"model_id": model_id, "dtype": str(dtype)}, sort_keys=True), flush=True)
    pipe = LTX2ImageToVideoPipeline.from_pretrained(model_id, torch_dtype=dtype)
    if config.get("enable_vae_tiling", True) and hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

    offload = str(config.get("cpu_offload", "sequential")).lower()
    if offload == "sequential":
        pipe.enable_sequential_cpu_offload(device="cuda")
    elif offload == "model":
        pipe.enable_model_cpu_offload(device="cuda")
    elif offload in {"false", "none", "no"}:
        pipe.to("cuda")
    else:
        raise ValueError(f"Unsupported cpu_offload: {offload}")

    width = int(config["width"])
    height = int(config["height"])
    num_frames = int(config["num_frames"])
    if width % 32 or height % 32:
        raise ValueError(f"width and height must be divisible by 32, got {width}x{height}")
    if (num_frames - 1) % 8:
        raise ValueError(f"num_frames must be 8k+1, got {num_frames}")

    image = Image.open(input_image_path).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    generator = torch.Generator("cuda").manual_seed(int(config["seed"]))
    total_steps = int(config["num_inference_steps"])
    started = time.time()
    video, audio = pipe(
        image=image,
        prompt=config["prompt"],
        negative_prompt=config.get("negative_prompt", ""),
        width=width,
        height=height,
        num_frames=num_frames,
        frame_rate=float(config["fps"]),
        num_inference_steps=total_steps,
        guidance_scale=float(config["guidance_scale"]),
        stg_scale=float(config.get("stg_scale", 1.0)),
        modality_scale=float(config.get("modality_scale", 1.0)),
        guidance_rescale=float(config.get("guidance_rescale", 0.0)),
        audio_guidance_scale=float(config.get("audio_guidance_scale", config["guidance_scale"])),
        audio_stg_scale=float(config.get("audio_stg_scale", 1.0)),
        audio_modality_scale=float(config.get("audio_modality_scale", 1.0)),
        audio_guidance_rescale=float(config.get("audio_guidance_rescale", 0.0)),
        spatio_temporal_guidance_blocks=[int(v) for v in config.get("spatio_temporal_guidance_blocks", [28])],
        use_cross_timestep=bool(config.get("use_cross_timestep", True)),
        generator=generator,
        output_type=config.get("output_type", "np"),
        return_dict=False,
    )
    generation_seconds = time.time() - started

    video_dir = output_dir / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    output_path = video_dir / config["output_filename"]
    encode_video(
        video[0],
        fps=float(config["fps"]),
        audio=audio[0].float().cpu(),
        audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
        output_path=str(output_path),
    )
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"LTX wrote empty output: {output_path}")

    return {
        "pipeline": "ltx2_i2v",
        "model_id": model_id,
        "output_video": str(output_path.relative_to(output_dir)),
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "fps": int(config["fps"]),
        "seed": int(config["seed"]),
        "num_inference_steps": total_steps,
        "generation_seconds": round(generation_seconds, 3),
        "video_bytes": output_path.stat().st_size,
    }


def _write_manifest(
    output_dir: Path,
    config: dict,
    config_uri: str,
    input_image_uri: str,
    output_uri: str,
    generation: dict,
) -> Path:
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_uri": config_uri,
        "input_image_uri": input_image_uri,
        "output_uri": output_uri,
        "prompt": config["prompt"],
        "negative_prompt": config.get("negative_prompt", ""),
        "generation": generation,
    }
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / config["manifest_filename"]
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _make_video_contact_sheet(video_path: Path, contact_sheet_path: Path) -> None:
    if not shutil.which("ffmpeg"):
        print("ffmpeg not available; skipping contact sheet generation.", flush=True)
        return
    contact_sheet_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        "fps=1,scale=320:-1,tile=5x1",
        str(contact_sheet_path),
    ]
    completed = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if completed.returncode != 0:
        print("Contact sheet generation failed; video output remains valid.", flush=True)
        print(completed.stdout, flush=True)


def _download_file(client, uri: str, destination: Path) -> None:
    bucket, blob = _parse_gcs_uri(uri)
    destination.parent.mkdir(parents=True, exist_ok=True)
    client.bucket(bucket).blob(blob).download_to_filename(destination)
    print(f"Downloaded {uri} to {destination}", flush=True)


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


def _read_yaml(path: Path) -> dict:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def _torch_dtype(value: str):
    import torch

    normalized = value.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {value}")


def _log_gpu_status() -> None:
    try:
        import torch

        payload = {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_runtime": torch.version.cuda,
        }
        if torch.cuda.is_available():
            payload["gpu_name"] = torch.cuda.get_device_name(0)
        print("AVW_VERTEX_GPU_STATUS=" + json.dumps(payload, sort_keys=True), flush=True)
    except Exception as exc:
        print(f"GPU status logging failed: {exc!r}", flush=True)
    try:
        subprocess.run(["nvidia-smi"], check=False)
    except FileNotFoundError:
        print("nvidia-smi not available", flush=True)


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
