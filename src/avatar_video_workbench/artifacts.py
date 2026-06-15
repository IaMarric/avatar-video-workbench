from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

from .config import WorkbenchError
from .datasets import image_paths, sha256_file


def build_dataset_manifest(images_dir: Path, *, trigger: str) -> list[dict[str, Any]]:
    images_dir = images_dir.expanduser().resolve()
    records: list[dict[str, Any]] = []
    for image_path in image_paths(images_dir):
        caption_path = image_path.with_suffix(".txt")
        caption = caption_path.read_text(encoding="utf-8").strip() if caption_path.exists() else ""
        with Image.open(image_path) as raw:
            image = ImageOps.exif_transpose(raw)
            width, height = image.size
        records.append(
            {
                "file_name": image_path.name,
                "caption_file": caption_path.name,
                "caption": caption,
                "trigger_present": trigger in caption,
                "width": width,
                "height": height,
                "aspect_ratio": round(width / height, 4),
                "sha256": sha256_file(image_path),
            }
        )
    return records


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def render_contact_sheet(
    images_dir: Path,
    out_path: Path,
    *,
    columns: int = 4,
    thumb_width: int = 160,
    thumb_height: int = 224,
) -> Path:
    images = image_paths(images_dir.expanduser().resolve())
    if not images:
        raise WorkbenchError(f"No images found for contact sheet: {images_dir}")
    columns = max(1, columns)
    rows = (len(images) + columns - 1) // columns
    label_height = 28
    padding = 10
    cell_width = thumb_width + padding * 2
    cell_height = thumb_height + label_height + padding * 2
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)

    for idx, image_path in enumerate(images):
        col = idx % columns
        row = idx // columns
        left = col * cell_width + padding
        top = row * cell_height + padding
        with Image.open(image_path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
            image.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        frame = Image.new("RGB", (thumb_width, thumb_height), (245, 245, 245))
        offset = ((thumb_width - image.width) // 2, (thumb_height - image.height) // 2)
        frame.paste(image, offset)
        sheet.paste(frame, (left, top))
        draw.text((left, top + thumb_height + 6), image_path.name[:24], fill=(40, 40, 40))

    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


def render_motion_storyboard(
    image_path: Path,
    out_path: Path,
    *,
    width: int = 512,
    height: int = 768,
    fps: int = 12,
    duration_seconds: float = 2.0,
    zoom: float = 1.08,
) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise WorkbenchError("ffmpeg is required to render storyboard video")
    image_path = image_path.expanduser().resolve()
    if not image_path.is_file():
        raise WorkbenchError(f"Storyboard source image not found: {image_path}")
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frame_count = max(2, int(round(fps * duration_seconds)))
    with tempfile.TemporaryDirectory(prefix="avw-storyboard-") as tmp:
        frames_dir = Path(tmp)
        with Image.open(image_path) as raw:
            base = ImageOps.exif_transpose(raw).convert("RGB")
        for index in range(frame_count):
            progress = index / max(frame_count - 1, 1)
            frame = _storyboard_frame(base, width=width, height=height, progress=progress, zoom=zoom)
            frame.save(frames_dir / f"frame_{index:04d}.png")
        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise WorkbenchError(completed.stderr.strip() or "ffmpeg failed to render storyboard video")
    return out_path


def _storyboard_frame(base: Image.Image, *, width: int, height: int, progress: float, zoom: float) -> Image.Image:
    target_ratio = width / height
    source_ratio = base.width / base.height
    if source_ratio > target_ratio:
        crop_h = base.height
        crop_w = int(crop_h * target_ratio)
    else:
        crop_w = base.width
        crop_h = int(crop_w / target_ratio)

    crop_w = max(1, int(crop_w / (1 + (zoom - 1) * progress)))
    crop_h = max(1, int(crop_h / (1 + (zoom - 1) * progress)))
    max_left = max(0, base.width - crop_w)
    max_top = max(0, base.height - crop_h)
    left = int(max_left * 0.5 + (progress - 0.5) * max_left * 0.16)
    top = int(max_top * 0.5 - (progress - 0.5) * max_top * 0.10)
    left = min(max(left, 0), max_left)
    top = min(max(top, 0), max_top)
    cropped = base.crop((left, top, left + crop_w, top + crop_h))
    return cropped.resize((width, height), Image.Resampling.LANCZOS)
