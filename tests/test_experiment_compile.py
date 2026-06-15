from __future__ import annotations

from pathlib import Path

import yaml
from PIL import Image

from avatar_video_workbench.experiments import CompileRunOptions, SmokeDemoOptions, compile_run, create_smoke_demo


def _write_image(path: Path, color: str = "white") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (80, 120), color).save(path)


def test_compile_run_creates_manifest_reports_and_prompts(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset" / "images"
    colors = ["white", "lightblue"]
    for index in range(2):
        stem = f"{index + 1:03d}"
        _write_image(dataset / f"{stem}.png", colors[index])
        (dataset / f"{stem}.txt").write_text(
            "demoavatar person, portrait, natural light\n",
            encoding="utf-8",
        )
    project_config = tmp_path / "avatar_project.yaml"
    project_config.write_text(
        """
project:
  name: demo-avatar
  trigger: demoavatar person
dataset:
  images_dir: dataset/images
  min_images: 2
""",
        encoding="utf-8",
    )

    manifest = compile_run(CompileRunOptions(project_config=project_config, out_dir=tmp_path / "compiled"))

    assert manifest["dataset"]["image_count"] == 2
    assert (tmp_path / "compiled" / "reports" / "dataset-validation.json").is_file()
    manifest_jsonl = tmp_path / "compiled" / "reports" / "dataset-manifest.jsonl"
    assert manifest_jsonl.is_file()
    assert len(manifest_jsonl.read_text(encoding="utf-8").splitlines()) == 2
    prompt_matrix = yaml.safe_load((tmp_path / "compiled" / "prompts" / "benchmark-prompts.yaml").read_text())
    assert len(prompt_matrix["prompts"]) >= 5
    assert all("demoavatar person" in item["prompt"] for item in prompt_matrix["prompts"])


def test_smoke_demo_runs_end_to_end(tmp_path: Path) -> None:
    manifest = create_smoke_demo(SmokeDemoOptions(out_dir=tmp_path / "smoke"))

    compiled = tmp_path / "smoke" / "compiled"
    assert manifest["dataset"]["image_count"] == 4
    assert (compiled / "manifest.json").is_file()
    assert (compiled / "prompts" / "benchmark-prompts.yaml").is_file()
    assert (compiled / "jobs" / "vertex-custom-job.yaml").is_file()
    assert (compiled / "reports" / "dataset-manifest.jsonl").is_file()
    assert (compiled / "previews" / "dataset-contact-sheet.png").stat().st_size > 0
    assert (compiled / "previews" / "motion-storyboard.mp4").stat().st_size > 0
