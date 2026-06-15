from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
PUBLICATION_PATH_PARTS = {".github", "docs", "src", "templates", "tests"}


@dataclass(frozen=True)
class DatasetValidationOptions:
    images_dir: Path
    trigger: str
    min_images: int = 1
    require_trigger: bool = True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_paths(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        return []
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_dataset(options: DatasetValidationOptions) -> dict[str, Any]:
    images_dir = options.images_dir.expanduser().resolve()
    problems: list[dict[str, str]] = []
    sizes: Counter[str] = Counter()
    caption_terms: Counter[str] = Counter()
    hashes: Counter[str] = Counter()
    checked_caption_count = 0
    trigger_caption_count = 0

    if not images_dir.exists():
        problems.append({"code": "missing_images_dir", "path": str(images_dir)})
        return _result(images_dir, options, [], [], sizes, caption_terms, hashes, problems)

    public_parts = PUBLICATION_PATH_PARTS.intersection(images_dir.parts)
    if public_parts and images_dir.name != ".gitkeep":
        problems.append(
            {
                "code": "dataset_in_publication_path",
                "path": str(images_dir),
                "path_part": sorted(public_parts)[0],
            }
        )

    images = image_paths(images_dir)
    captions = sorted(path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt")

    if len(images) < options.min_images:
        problems.append({"code": "not_enough_images", "count": str(len(images)), "minimum": str(options.min_images)})

    for image_path in images:
        caption_path = image_path.with_suffix(".txt")
        if not caption_path.exists():
            problems.append({"code": "missing_caption", "file": image_path.name})
            continue

        caption = caption_path.read_text(encoding="utf-8").strip()
        if not caption:
            problems.append({"code": "empty_caption", "file": caption_path.name})
        if caption:
            checked_caption_count += 1
            if options.trigger in caption:
                trigger_caption_count += 1
            elif options.require_trigger:
                problems.append({"code": "missing_trigger", "file": caption_path.name})

        for term in _caption_terms(caption, options.trigger):
            caption_terms[term] += 1

        try:
            with Image.open(image_path) as raw:
                image = ImageOps.exif_transpose(raw)
                sizes[f"{image.width}x{image.height}"] += 1
        except Exception as exc:  # pragma: no cover - Pillow error types vary by decoder
            problems.append({"code": "unreadable_image", "file": image_path.name, "error": str(exc)})

        hashes[sha256_file(image_path)] += 1

    image_stems = {path.stem for path in images}
    for caption_path in captions:
        if caption_path.stem not in image_stems:
            problems.append({"code": "orphan_caption", "file": caption_path.name})

    for digest, count in hashes.items():
        if count > 1:
            problems.append({"code": "duplicate_image_hash", "sha256": digest, "count": str(count)})

    if options.require_trigger and checked_caption_count and trigger_caption_count != checked_caption_count:
        problems.append(
            {
                "code": "trigger_coverage_incomplete",
                "captions_with_trigger": str(trigger_caption_count),
                "caption_count": str(checked_caption_count),
            }
        )

    return _result(images_dir, options, images, captions, sizes, caption_terms, hashes, problems)


def _caption_terms(caption: str, trigger: str) -> list[str]:
    terms = []
    lowered = caption.lower()
    for candidate in [
        trigger.lower(),
        "portrait",
        "full body",
        "selfie",
        "phone photo",
        "indoor",
        "outdoor",
        "natural light",
        "low light",
        "walking",
        "sitting",
    ]:
        if candidate and candidate in lowered:
            terms.append(candidate)
    return terms


def _result(
    images_dir: Path,
    options: DatasetValidationOptions,
    images: list[Path],
    captions: list[Path],
    sizes: Counter[str],
    caption_terms: Counter[str],
    hashes: Counter[str],
    problems: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "images_dir": str(images_dir),
        "trigger": options.trigger,
        "image_count": len(images),
        "caption_count": len(captions),
        "trigger_caption_count": sum(1 for path in captions if options.trigger in path.read_text(encoding="utf-8", errors="replace")),
        "unique_image_hashes": len(hashes),
        "sizes": dict(sizes.most_common()),
        "caption_terms": dict(caption_terms.most_common()),
        "problems": problems,
        "ok": not problems and len(images) == len(captions) and len(images) >= options.min_images,
    }
