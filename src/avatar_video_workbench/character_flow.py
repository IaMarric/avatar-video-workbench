from __future__ import annotations

import base64
import io
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image, ImageOps

from .artifacts import render_contact_sheet
from .config import WorkbenchError, write_json, write_yaml
from .datasets import sha256_file

DEFAULT_IMAGE_MODEL = "gemini-3.1-flash-image"


@dataclass(frozen=True)
class CharacterDatasetOptions:
    source_image: Path
    out_dir: Path
    trigger: str
    theme: str = "pirate"
    variant_count: int = 10
    model: str = DEFAULT_IMAGE_MODEL
    vertex_project: str | None = None
    vertex_location: str | None = None
    auth_mode: str = "sdk"
    plan_only: bool = False
    ltx_model_path: str | None = None
    ltx_text_encoder_path: str | None = None
    training_steps: int = 1200
    resolution_bucket: str = "512x512x1"


ImageGenerator = Callable[[Path, str, int], Image.Image]


def create_character_dataset(
    options: CharacterDatasetOptions,
    *,
    image_generator: ImageGenerator | None = None,
) -> dict:
    source_image = options.source_image.expanduser().resolve()
    if not source_image.is_file():
        raise WorkbenchError(f"Source image not found: {source_image}")
    if options.variant_count < 1:
        raise WorkbenchError("--variant-count must be at least 1")
    if not options.trigger.strip():
        raise WorkbenchError("--trigger is required")

    preset = _theme_preset(options.theme)
    out_dir = options.out_dir.expanduser().resolve()
    source_dir = out_dir / "source"
    dataset_dir = out_dir / "dataset" / "images"
    reports_dir = out_dir / "reports"
    trainer_dir = out_dir / "ltx_trainer"
    configs_dir = out_dir / "configs"
    for path in [source_dir, dataset_dir, reports_dir, trainer_dir, configs_dir]:
        path.mkdir(parents=True, exist_ok=True)

    source_copy = source_dir / ("source" + source_image.suffix.lower())
    _copy_normalized_image(source_image, source_copy)

    prompts = build_variant_prompts(
        trigger=options.trigger,
        theme=options.theme,
        variant_count=options.variant_count,
    )
    requests_path = reports_dir / "nano-banana-requests.jsonl"
    with requests_path.open("w", encoding="utf-8") as handle:
        for prompt in prompts:
            handle.write(json.dumps(prompt, ensure_ascii=False, sort_keys=True) + "\n")

    generated: list[dict] = []
    generator = image_generator
    if generator is None and not options.plan_only:
        generator = _gemini_image_generator(
            model=options.model,
            vertex_project=options.vertex_project,
            vertex_location=options.vertex_location,
            auth_mode=options.auth_mode,
        )

    for prompt in prompts:
        stem = f"{prompt['index']:03d}"
        image_path: Path | None = dataset_dir / f"{stem}.png"
        caption_path = dataset_dir / f"{stem}.txt"
        caption = _caption_for_variant(options.trigger, preset["caption"], prompt["variant"])
        if image_path is not None and image_path.exists():
            pass
        elif generator is not None:
            image = generator(source_copy, prompt["prompt"], prompt["index"])
            _save_training_image(image, image_path)
        elif not image_path.exists():
            image_path = None
        caption_path.write_text(caption + "\n", encoding="utf-8")
        generated.append(
            {
                "index": prompt["index"],
                "image": image_path.relative_to(out_dir).as_posix() if image_path is not None else None,
                "caption": caption_path.relative_to(out_dir).as_posix(),
                "variant": prompt["variant"],
                "prompt_id": prompt["id"],
            }
        )

    dataset_json_path = trainer_dir / "dataset.json"
    trainer_rows = []
    for item in generated:
        if not item["image"]:
            continue
        trainer_rows.append(
            {
                "media_path": "../" + item["image"],
                "caption": Path(out_dir / item["caption"]).read_text(encoding="utf-8").strip(),
            }
        )
    write_json(dataset_json_path, trainer_rows)

    contact_sheet_path = None
    if trainer_rows:
        contact_sheet_path = render_contact_sheet(dataset_dir, reports_dir / "dataset-contact-sheet.png", columns=5)

    training_config_path = None
    if options.ltx_model_path and options.ltx_text_encoder_path:
        training_config_path = configs_dir / "ltx_lora_training.yaml"
        write_yaml(
            training_config_path,
            build_ltx_lora_training_config(
                model_path=options.ltx_model_path,
                text_encoder_path=options.ltx_text_encoder_path,
                preprocessed_data_root=".precomputed",
                output_dir="outputs/ltx_lora",
                validation_prompt=preset["video_prompt"].format(trigger=options.trigger),
                steps=options.training_steps,
            ),
        )

    manifest = {
        "schema_version": "1.0",
        "source_image": source_copy.relative_to(out_dir).as_posix(),
        "source_sha256": sha256_file(source_copy),
        "theme": options.theme,
        "trigger": options.trigger,
        "image_model": options.model,
        "variant_count": options.variant_count,
        "plan_only": options.plan_only,
        "dataset_dir": dataset_dir.relative_to(out_dir).as_posix(),
        "nano_banana_requests": requests_path.relative_to(out_dir).as_posix(),
        "ltx_trainer_dataset": dataset_json_path.relative_to(out_dir).as_posix(),
        "ltx_training_config": training_config_path.relative_to(out_dir).as_posix() if training_config_path else None,
        "contact_sheet": contact_sheet_path.relative_to(out_dir).as_posix() if contact_sheet_path else None,
        "generated": generated,
        "next_steps": [
            "Review generated variants and remove weak images before training.",
            "Run LTX trainer preprocessing on ltx_trainer/dataset.json with the same trigger.",
            "Train an LTX LoRA using the generated config or submit-ltx-lora-train.",
            "Run submit-ltx-i2v with the trained LoRA URI to generate video.",
        ],
    }
    manifest_path = out_dir / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest


def build_variant_prompts(*, trigger: str, theme: str, variant_count: int) -> list[dict]:
    preset = _theme_preset(theme)
    prompts = []
    variants = preset["variants"]
    for index in range(1, variant_count + 1):
        variant = variants[(index - 1) % len(variants)]
        prompts.append(
            {
                "id": f"{theme}-{index:03d}",
                "index": index,
                "variant": variant,
                "prompt": preset["edit_prompt"].format(trigger=trigger, variant=variant),
            }
        )
    return prompts


def build_ltx_lora_training_config(
    *,
    model_path: str,
    text_encoder_path: str,
    preprocessed_data_root: str,
    output_dir: str,
    validation_prompt: str,
    steps: int,
) -> dict:
    return {
        "model": {
            "model_path": model_path,
            "text_encoder_path": text_encoder_path,
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
        "data": {
            "preprocessed_data_root": preprocessed_data_root,
            "num_dataloader_workers": 2,
        },
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
        "checkpoints": {
            "interval": 250,
            "keep_last_n": 3,
            "precision": "bfloat16",
        },
        "flow_matching": {
            "timestep_sampling_mode": "shifted_logit_normal",
            "timestep_sampling_params": {},
        },
        "hub": {"push_to_hub": False, "hub_model_id": None},
        "wandb": {
            "enabled": False,
            "project": "avatar-video-workbench",
            "entity": None,
            "tags": ["ltx2", "lora", "avatar-video-workbench"],
            "log_validation_videos": False,
        },
        "seed": 42,
        "output_dir": output_dir,
    }


def _gemini_image_generator(
    *,
    model: str,
    vertex_project: str | None,
    vertex_location: str | None,
    auth_mode: str,
) -> ImageGenerator:
    normalized_auth_mode = auth_mode.lower().strip()
    if normalized_auth_mode == "gcloud":
        if not vertex_project or not vertex_location:
            raise WorkbenchError("--auth-mode gcloud requires --vertex-project and --vertex-location")
        return _gcloud_vertex_image_generator(model=model, vertex_project=vertex_project, vertex_location=vertex_location)
    if normalized_auth_mode != "sdk":
        raise WorkbenchError("--auth-mode must be sdk or gcloud")

    try:
        from google import genai
    except Exception as exc:
        raise WorkbenchError("Install google-genai to generate Nano Banana images: python -m pip install google-genai") from exc

    if vertex_project or vertex_location:
        if not vertex_project or not vertex_location:
            raise WorkbenchError("--vertex-project and --vertex-location must be provided together")
        client = genai.Client(vertexai=True, project=vertex_project, location=vertex_location)
    else:
        client = genai.Client()

    def generate(source_image: Path, prompt: str, index: int) -> Image.Image:
        with Image.open(source_image) as raw:
            source = ImageOps.exif_transpose(raw).convert("RGB")
        try:
            response = client.models.generate_content(model=model, contents=[prompt, source])
        except Exception as exc:
            raise WorkbenchError(f"Nano Banana generation failed for variant {index}: {exc}") from exc
        for part in response.parts:
            image = part.as_image() if hasattr(part, "as_image") else None
            if image is not None:
                return image.convert("RGB")
            if getattr(part, "inline_data", None) is not None and hasattr(part, "as_image"):
                return part.as_image().convert("RGB")
        raise WorkbenchError(f"Nano Banana response {index} did not include an image")

    return generate


def _gcloud_vertex_image_generator(*, model: str, vertex_project: str, vertex_location: str) -> ImageGenerator:
    import requests

    endpoint = (
        "https://aiplatform.googleapis.com"
        f"/v1/projects/{vertex_project}/locations/{vertex_location}/publishers/google/models/{model}:generateContent"
    )

    def generate(source_image: Path, prompt: str, index: int) -> Image.Image:
        token = _gcloud_access_token()
        image_bytes = source_image.read_bytes()
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": _mime_type(source_image),
                                "data": base64.b64encode(image_bytes).decode("ascii"),
                            }
                        },
                    ],
                }
            ]
        }
        response = None
        for attempt in range(1, 6):
            response = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=300,
            )
            if response.status_code not in {429, 500, 502, 503, 504}:
                break
            wait_seconds = min(90, 12 * attempt)
            print(
                f"Nano Banana Vertex returned {response.status_code} for variant {index}; retrying in {wait_seconds}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(wait_seconds)
            token = _gcloud_access_token()
        if response is None or response.status_code >= 400:
            status = response.status_code if response is not None else "no_response"
            text = response.text if response is not None else ""
            raise WorkbenchError(f"Nano Banana Vertex REST failed for variant {index}: {status} {text}")
        data = response.json()
        image_data: str | None = None
        for candidate in data.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                inline = part.get("inlineData") or part.get("inline_data")
                if inline and inline.get("data"):
                    image_data = inline["data"]
        if not image_data:
            raise WorkbenchError(f"Nano Banana Vertex REST response {index} did not include an image")
        return Image.open(io.BytesIO(base64.b64decode(image_data))).convert("RGB")

    return generate


def _gcloud_access_token() -> str:
    completed = subprocess.run(["gcloud", "auth", "print-access-token"], text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise WorkbenchError(completed.stderr.strip() or "gcloud auth print-access-token failed")
    token = completed.stdout.strip()
    if not token:
        raise WorkbenchError("gcloud auth print-access-token returned an empty token")
    return token


def _mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"


def _theme_preset(theme: str) -> dict:
    key = theme.lower().strip()
    presets = {
        "pirate": {
            "caption": "cinematic pirate portrait, adventurous costume, consistent face, natural skin texture",
            "edit_prompt": (
                "Using the uploaded reference photo, create a training image of the same authorized person as {trigger}. "
                "Transform the outfit and setting into a family-friendly cinematic pirate character: {variant}. "
                "Preserve facial identity, age, skin tone, eye shape, and natural proportions. "
                "Keep the image realistic, family-friendly, no gore, no text, no watermark, single subject, sharp face."
            ),
            "video_prompt": (
                "{trigger} stands on the deck of a wooden ship at golden hour, wearing a tasteful pirate coat and sash. "
                "The camera makes a slow handheld push-in while sea wind subtly moves the fabric and hair. "
                "The character gives a small confident smile, identity remains stable, warm cinematic light, no text."
            ),
            "variants": [
                "weathered captain coat, brass buttons, warm sunset deck",
                "navy blue coat, red sash, soft cloudy daylight",
                "cream shirt, leather vest, harbor background",
                "dark green captain jacket, rope rigging behind",
                "burgundy coat, simple tricorn hat, calm sea horizon",
                "practical explorer outfit, rolled map in hand",
                "black coat with gold trim, lantern-lit cabin",
                "brown leather vest, ship wheel nearby",
                "stormy sky, coat moving in wind, serious expression",
                "bright tropical harbor, relaxed friendly expression",
            ],
        }
    }
    if key not in presets:
        raise WorkbenchError(f"Unknown theme: {theme}. Available themes: {', '.join(sorted(presets))}")
    return presets[key]


def _caption_for_variant(trigger: str, base_caption: str, variant: str) -> str:
    return f"{trigger}, {base_caption}, {variant}"


def _copy_normalized_image(source: Path, destination: Path) -> None:
    with Image.open(source) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def _save_training_image(image: Image.Image, destination: Path) -> None:
    normalized = ImageOps.exif_transpose(image).convert("RGB")
    normalized.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", normalized.size, (255, 255, 255))
    canvas.paste(normalized)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)
