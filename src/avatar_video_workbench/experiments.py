from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .config import WorkbenchError, load_mapping, require_keys, write_json, write_yaml
from .datasets import DatasetValidationOptions, validate_dataset
from .vertex import render_vertex_job


DEFAULT_NEGATIVE_PROMPT = "cartoon, CGI, plastic skin, distorted face, extra fingers, text, watermark"


@dataclass(frozen=True)
class CompileRunOptions:
    project_config: Path
    out_dir: Path
    vertex_config: Path | None = None
    vertex_template: Path | None = None


@dataclass(frozen=True)
class SmokeDemoOptions:
    out_dir: Path
    force: bool = False


def compile_run(options: CompileRunOptions) -> dict[str, Any]:
    project_config = options.project_config.expanduser().resolve()
    out_dir = options.out_dir.expanduser().resolve()
    data = load_mapping(project_config)
    project = _mapping(data.get("project"), "project")
    dataset = _mapping(data.get("dataset"), "dataset")
    require_keys(project, ["name", "trigger"], label="project")
    require_keys(dataset, ["images_dir"], label="dataset")

    trigger = str(project["trigger"])
    run_id = _slug(str(project.get("run_id") or project["name"]))
    images_dir = _resolve_path(str(dataset["images_dir"]), base_dir=project_config.parent)
    min_images = int(dataset.get("min_images") or 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = out_dir / "reports"
    prompts_dir = out_dir / "prompts"
    jobs_dir = out_dir / "jobs"
    reports_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir.mkdir(parents=True, exist_ok=True)

    validation = validate_dataset(
        DatasetValidationOptions(
            images_dir=images_dir,
            trigger=trigger,
            min_images=min_images,
            require_trigger=bool(dataset.get("require_trigger", True)),
        )
    )
    validation_path = reports_dir / "dataset-validation.json"
    write_json(validation_path, validation)
    if not validation["ok"]:
        raise WorkbenchError(f"Dataset validation failed; see {validation_path}")

    prompt_matrix = build_prompt_matrix(project, data.get("benchmarks") or {})
    prompt_matrix_path = prompts_dir / "benchmark-prompts.yaml"
    write_yaml(prompt_matrix_path, {"prompts": prompt_matrix})

    vertex_job_path: Path | None = None
    if options.vertex_config or options.vertex_template:
        if not options.vertex_config or not options.vertex_template:
            raise WorkbenchError("--vertex-config and --vertex-template must be provided together")
        vertex_job_path = jobs_dir / "vertex-custom-job.yaml"
        vertex_job_path.write_text(
            render_vertex_job(options.vertex_config.expanduser().resolve(), options.vertex_template.expanduser().resolve()),
            encoding="utf-8",
        )

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "project": {
            "name": project["name"],
            "trigger": trigger,
            "character_policy": project.get("character_policy", "fictional_or_authorized_public_demo"),
        },
        "dataset": {
            "images_dir": str(images_dir),
            "image_count": validation["image_count"],
            "caption_count": validation["caption_count"],
            "unique_image_hashes": validation["unique_image_hashes"],
            "validation_report": _relative_to(validation_path, out_dir),
        },
        "artifacts": {
            "prompt_matrix": _relative_to(prompt_matrix_path, out_dir),
            "vertex_job": _relative_to(vertex_job_path, out_dir) if vertex_job_path else None,
        },
        "next_steps": [
            "Review reports/dataset-validation.json.",
            "Run still-image benchmarks from prompts/benchmark-prompts.yaml.",
            "Select one or more hero stills outside git.",
            "Render provider-specific image-to-video jobs from tracked configs.",
            "Keep generated media, datasets, and LoRA checkpoints out of the repository.",
        ],
    }
    manifest_path = out_dir / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest


def build_prompt_matrix(project: dict[str, Any], benchmark_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    trigger = str(project["trigger"])
    negative_prompt = str(benchmark_cfg.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT)
    prompt_specs = benchmark_cfg.get("prompts")
    if prompt_specs is None:
        prompt_specs = _default_prompt_specs()
    if not isinstance(prompt_specs, list):
        raise WorkbenchError("benchmarks.prompts must be a list")

    prompts: list[dict[str, Any]] = []
    for index, item in enumerate(prompt_specs, start=1):
        if not isinstance(item, dict):
            raise WorkbenchError(f"benchmarks.prompts[{index}] must be a mapping")
        require_keys(item, ["id", "stage", "prompt"], label=f"benchmarks.prompts[{index}]")
        rendered = str(item["prompt"]).format(trigger=trigger)
        prompts.append(
            {
                "id": item["id"],
                "stage": item["stage"],
                "prompt": rendered,
                "negative_prompt": item.get("negative_prompt", negative_prompt),
                "width": int(item.get("width", 768)),
                "height": int(item.get("height", 1152)),
                "seed": int(item.get("seed", 42 + index)),
                "duration_seconds": item.get("duration_seconds"),
                "notes": item.get("notes", ""),
            }
        )
    return prompts


def create_smoke_demo(options: SmokeDemoOptions) -> dict[str, Any]:
    out_dir = options.out_dir.expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()) and not options.force:
        raise WorkbenchError(f"Smoke demo output already exists and is not empty: {out_dir}")

    dataset_dir = out_dir / "dataset" / "images"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    trigger = "demoavatar person"
    captions = [
        "portrait, natural light, neutral studio background",
        "full-length standing photo, outdoor walkway, soft daylight",
        "phone photo, indoor cafe table, candid framing",
        "walking pose, city street, golden hour",
    ]
    for idx, caption in enumerate(captions, start=1):
        stem = f"{idx:03d}"
        _write_demo_image(dataset_dir / f"{stem}.png", idx)
        (dataset_dir / f"{stem}.txt").write_text(f"{trigger}, {caption}\n", encoding="utf-8")

    project_config = out_dir / "avatar_project.yaml"
    write_yaml(
        project_config,
        {
            "project": {
                "name": "demo-avatar",
                "trigger": trigger,
                "character_policy": "fictional_or_authorized_public_demo",
            },
            "dataset": {
                "images_dir": "dataset/images",
                "min_images": len(captions),
            },
        },
    )

    vertex_config = out_dir / "ltx_i2v.yaml"
    write_yaml(
        vertex_config,
        {
            "run": {
                "display_name": "demo-avatar-ltx-i2v",
                "id": "demo-avatar-ltx-i2v",
            },
            "vertex": {
                "machine_type": "a3-highgpu-1g",
                "accelerator_type": "NVIDIA_H100_80GB",
                "accelerator_count": 1,
                "boot_disk_size_gb": 1000,
            },
            "container": {
                "image_uri": "us-central1-docker.pkg.dev/demo-project/avatar-video-runtime/runner:latest",
                "command": "python -m avatar_video_runtime.ltx_i2v",
            },
            "env": {
                "CONFIG_URI": "/workspace/config/i2v.yaml",
                "INPUT_IMAGE_URI": "/workspace/input/hero.png",
                "OUTPUT_URI": "/workspace/output",
                "HF_HOME": "/workspace/.cache/huggingface",
            },
        },
    )

    vertex_template = out_dir / "vertex_template.yaml"
    vertex_template.write_text(
        """displayName: ${display_name}
jobSpec:
  workerPoolSpecs:
    - machineSpec:
        machineType: ${machine_type}
        acceleratorType: ${accelerator_type}
        acceleratorCount: ${accelerator_count}
      diskSpec:
        bootDiskType: pd-ssd
        bootDiskSizeGb: ${boot_disk_size_gb}
      replicaCount: 1
      containerSpec:
        imageUri: ${image_uri}
        command:
          - bash
          - -lc
        args:
          - ${command}
        env:
          - name: CONFIG_URI
            value: ${env_CONFIG_URI}
          - name: INPUT_IMAGE_URI
            value: ${env_INPUT_IMAGE_URI}
          - name: OUTPUT_URI
            value: ${env_OUTPUT_URI}
          - name: HF_HOME
            value: ${env_HF_HOME}
""",
        encoding="utf-8",
    )

    return compile_run(
        CompileRunOptions(
            project_config=project_config,
            out_dir=out_dir / "compiled",
            vertex_config=vertex_config,
            vertex_template=vertex_template,
        )
    )


def _default_prompt_specs() -> list[dict[str, Any]]:
    return [
        {
            "id": "still-portrait-natural-light",
            "stage": "still",
            "prompt": "{trigger}, realistic portrait photo, natural light, neutral background, sharp eyes",
            "notes": "Identity anchor for face consistency.",
        },
        {
            "id": "still-full-length-daylight",
            "stage": "still",
            "prompt": "{trigger}, realistic full-length photo, relaxed standing pose, daylight street scene",
            "notes": "Checks proportions and wardrobe drift.",
        },
        {
            "id": "still-phone-candid-indoor",
            "stage": "still",
            "prompt": "{trigger}, candid phone photo, seated at a cafe table, indoor ambient light",
            "notes": "Checks casual UGC-style framing.",
        },
        {
            "id": "i2v-walking-golden-hour",
            "stage": "image_to_video",
            "prompt": "{trigger}, handheld phone video, walking through a quiet city street at golden hour",
            "duration_seconds": 4,
            "notes": "Motion baseline for identity retention.",
        },
        {
            "id": "i2v-seated-camera-turn",
            "stage": "image_to_video",
            "prompt": "{trigger}, subtle camera move around a seated avatar, soft daylight, natural expression",
            "duration_seconds": 4,
            "notes": "Low-motion comparison route.",
        },
    ]


def _write_demo_image(path: Path, index: int) -> None:
    image = Image.new("RGB", (256, 384), (232, 235, 229))
    draw = ImageDraw.Draw(image)
    accent = [(72, 111, 147), (146, 86, 78), (82, 130, 99), (121, 95, 151)][(index - 1) % 4]
    draw.rectangle((0, 250, 256, 384), fill=(210, 214, 207))
    draw.ellipse((82, 54, 174, 146), fill=(226, 190, 160), outline=(70, 70, 70), width=3)
    draw.rectangle((72, 150, 184, 292), fill=accent, outline=(70, 70, 70), width=3)
    draw.line((105, 94, 115, 94), fill=(40, 40, 40), width=3)
    draw.line((141, 94, 151, 94), fill=(40, 40, 40), width=3)
    draw.arc((110, 105, 146, 130), start=15, end=165, fill=(90, 55, 55), width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WorkbenchError(f"{label} must be a mapping")
    return value


def _resolve_path(value: str, *, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _relative_to(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip().lower()).strip("-")
    return slug or "avatar-run"
