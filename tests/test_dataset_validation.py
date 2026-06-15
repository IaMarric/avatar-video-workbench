from __future__ import annotations

from pathlib import Path

from PIL import Image

from avatar_video_workbench.datasets import DatasetValidationOptions, validate_dataset


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 96), "white").save(path)


def test_validate_dataset_accepts_paired_images(tmp_path: Path) -> None:
    images = tmp_path / "images"
    _write_image(images / "001.png")
    (images / "001.txt").write_text("demoavatar person, portrait, natural light\n", encoding="utf-8")

    result = validate_dataset(DatasetValidationOptions(images_dir=images, trigger="demoavatar person"))

    assert result["ok"] is True
    assert result["image_count"] == 1
    assert result["caption_count"] == 1
    assert result["sizes"] == {"64x96": 1}


def test_validate_dataset_reports_missing_caption(tmp_path: Path) -> None:
    images = tmp_path / "images"
    _write_image(images / "001.png")

    result = validate_dataset(DatasetValidationOptions(images_dir=images, trigger="demoavatar person"))

    assert result["ok"] is False
    assert {"code": "missing_caption", "file": "001.png"} in result["problems"]


def test_validate_dataset_reports_trigger_coverage(tmp_path: Path) -> None:
    images = tmp_path / "images"
    _write_image(images / "001.png")
    (images / "001.txt").write_text("demoavatar person, portrait\n", encoding="utf-8")
    _write_image(images / "002.png")
    (images / "002.txt").write_text("portrait without trigger\n", encoding="utf-8")

    result = validate_dataset(DatasetValidationOptions(images_dir=images, trigger="demoavatar person"))

    assert result["ok"] is False
    assert result["trigger_caption_count"] == 1
    assert {"code": "missing_trigger", "file": "002.txt"} in result["problems"]
    assert {
        "code": "trigger_coverage_incomplete",
        "captions_with_trigger": "1",
        "caption_count": "2",
    } in result["problems"]


def test_validate_dataset_flags_publication_paths(tmp_path: Path) -> None:
    images = tmp_path / "docs" / "images"
    _write_image(images / "001.png")
    (images / "001.txt").write_text("demoavatar person, portrait\n", encoding="utf-8")

    result = validate_dataset(DatasetValidationOptions(images_dir=images, trigger="demoavatar person"))

    assert result["ok"] is False
    assert any(problem["code"] == "dataset_in_publication_path" for problem in result["problems"])
