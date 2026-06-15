from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import WorkbenchError, load_mapping, write_json


def export_backend_metadata(input_path: Path) -> dict[str, Any]:
    """Export public-safe backend metadata from an AVW run config or manifest."""
    input_path = input_path.expanduser().resolve()
    payload = load_mapping(input_path)

    if _is_ltx_i2v_runtime_manifest(payload):
        return _metadata_from_ltx_i2v_runtime_manifest(payload)
    if _is_ltx_i2v_config(payload):
        return _metadata_from_ltx_i2v_config(payload, source_type="ltx_i2v_config", provider=None)
    if _is_ltx_i2v_submission_manifest(payload):
        config_path = input_path.parent / "config" / "ltx_i2v.yaml"
        if not config_path.is_file():
            raise WorkbenchError(
                "LTX submission manifests do not contain prompt/model metadata; "
                f"pass the staged config directly or keep {config_path}"
            )
        config = load_mapping(config_path)
        if not _is_ltx_i2v_config(config):
            raise WorkbenchError(f"Expected an LTX I2V config at {config_path}")
        return _metadata_from_ltx_i2v_config(
            config,
            source_type="ltx_i2v_submission_manifest",
            provider="vertex-ai",
        )
    raise WorkbenchError(
        "Unsupported backend metadata input. Expected an LTX I2V config, "
        "an LTX I2V runtime manifest, or a submission manifest with a sibling config/ltx_i2v.yaml."
    )


def write_backend_metadata(input_path: Path, out_path: Path) -> dict[str, Any]:
    metadata = export_backend_metadata(input_path)
    write_json(out_path.expanduser().resolve(), metadata)
    return metadata


def _metadata_from_ltx_i2v_config(
    config: dict[str, Any],
    *,
    source_type: str,
    provider: str | None,
) -> dict[str, Any]:
    lora_ref = config.get("lora_weights_uri") or config.get("lora_weights_path")
    metadata = _normalized_metadata(
        source_type=source_type,
        provider=provider,
        backend_name="ltx2_i2v",
        runtime="diffusers",
        model_id=_required_str(config, "model_id"),
        prompt=_required_str(config, "prompt"),
        negative_prompt=str(config.get("negative_prompt") or ""),
        width=_required_int(config, "width"),
        height=_required_int(config, "height"),
        frames=_required_int(config, "num_frames"),
        fps=_required_float(config, "fps"),
        seed=_required_int(config, "seed"),
        steps=_required_int(config, "num_inference_steps"),
        guidance_scale=_optional_float(config.get("guidance_scale")),
        lora_ref=lora_ref,
        lora_scale=_optional_float(config.get("lora_scale")) or 1.0,
        runtime_metrics={},
    )
    return metadata


def _metadata_from_ltx_i2v_runtime_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    generation = _required_mapping(manifest, "generation")
    runtime_metrics = {
        key: generation[key]
        for key in ["generation_seconds", "video_bytes"]
        if key in generation and generation[key] is not None
    }
    metadata = _normalized_metadata(
        source_type="ltx_i2v_runtime_manifest",
        provider="vertex-ai",
        backend_name=str(generation.get("pipeline") or "ltx2_i2v"),
        runtime="diffusers",
        model_id=_required_str(generation, "model_id"),
        prompt=_required_str(manifest, "prompt"),
        negative_prompt=str(manifest.get("negative_prompt") or ""),
        width=_required_int(generation, "width"),
        height=_required_int(generation, "height"),
        frames=_required_int(generation, "num_frames"),
        fps=_required_float(generation, "fps"),
        seed=_required_int(generation, "seed"),
        steps=_required_int(generation, "num_inference_steps"),
        guidance_scale=_optional_float(generation.get("guidance_scale")),
        lora_ref=generation.get("lora_weights_path") or manifest.get("lora_weights_uri"),
        lora_scale=_optional_float(generation.get("lora_scale")) or 1.0,
        runtime_metrics=runtime_metrics,
    )
    return metadata


def _normalized_metadata(
    *,
    source_type: str,
    provider: str | None,
    backend_name: str,
    runtime: str,
    model_id: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    frames: int,
    fps: float,
    seed: int,
    steps: int,
    guidance_scale: float | None,
    lora_ref: Any,
    lora_scale: float,
    runtime_metrics: dict[str, Any],
) -> dict[str, Any]:
    lora_enabled = bool(lora_ref)
    sampling = {
        "seed": seed,
        "steps": steps,
    }
    if guidance_scale is not None:
        sampling["guidance_scale"] = guidance_scale

    video = {
        "width": width,
        "height": height,
        "frames": frames,
        "fps": int(fps) if float(fps).is_integer() else fps,
        "duration_seconds": round(frames / fps, 4) if fps else None,
    }

    metadata: dict[str, Any] = {
        "schema_version": "1.0",
        "source": {
            "tool": "avatar-video-workbench",
            "source_type": source_type,
        },
        "backend": {
            "name": backend_name,
            "runtime": runtime,
        },
        "model": {
            "id": model_id,
        },
        "prompt": {
            "positive": prompt,
            "negative": negative_prompt,
        },
        "video": _drop_none(video),
        "sampling": sampling,
        "lora": {
            "enabled": lora_enabled,
            "scale": lora_scale if lora_enabled else None,
            "artifact": {
                "kind": "lora_weights",
                "reference": "private_external_artifact",
            }
            if lora_enabled
            else None,
        },
        "runtime_metrics": runtime_metrics,
        "comfyui_comparison": _drop_none(
            {
                "model": model_id,
                "positive_prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "frames": frames,
                "fps": int(fps) if float(fps).is_integer() else fps,
                "seed": seed,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "lora_enabled": lora_enabled,
                "lora_scale": lora_scale if lora_enabled else None,
            }
        ),
        "privacy": {
            "generated_media_omitted": True,
            "private_paths_omitted": True,
            "external_artifact_uris_omitted": True,
        },
    }
    if provider:
        metadata["backend"]["provider"] = provider
    return _drop_none(metadata)


def _is_ltx_i2v_config(payload: dict[str, Any]) -> bool:
    required = {"model_id", "prompt", "width", "height", "num_frames", "fps", "seed", "num_inference_steps"}
    return required.issubset(payload.keys())


def _is_ltx_i2v_runtime_manifest(payload: dict[str, Any]) -> bool:
    return "generation" in payload and "prompt" in payload


def _is_ltx_i2v_submission_manifest(payload: dict[str, Any]) -> bool:
    return "run_id" in payload and "vertex_job_yaml" in payload and "gcs" in payload


def _required_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise WorkbenchError(f"Expected mapping field: {key}")
    return value


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if value in (None, ""):
        raise WorkbenchError(f"Missing required field: {key}")
    return str(value)


def _required_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if value in (None, ""):
        raise WorkbenchError(f"Missing required field: {key}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise WorkbenchError(f"Expected integer field {key}, got {value!r}") from exc


def _required_float(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if value in (None, ""):
        raise WorkbenchError(f"Missing required field: {key}")
    converted = _optional_float(value)
    if converted is None:
        raise WorkbenchError(f"Expected numeric field {key}, got {value!r}")
    return converted


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise WorkbenchError(f"Expected numeric value, got {value!r}") from exc


def _drop_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _drop_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_drop_none(item) for item in value if item is not None]
    return value
