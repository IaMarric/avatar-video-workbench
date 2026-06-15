from __future__ import annotations

from pathlib import Path

from avatar_video_workbench.vertex import render_vertex_job


def test_render_vertex_job_from_template(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    template = tmp_path / "template.yaml"
    config.write_text(
        """
run:
  display_name: demo-job
vertex:
  machine_type: a3-highgpu-1g
  accelerator_type: NVIDIA_H100_80GB
  accelerator_count: 1
  boot_disk_size_gb: 1000
container:
  image_uri: REGION-docker.pkg.dev/PROJECT/REPOSITORY/image:latest
  command: python demo.py
env:
  CONFIG_URI: gs://YOUR_BUCKET/config.yaml
  INPUT_IMAGE_URI: gs://YOUR_BUCKET/input.png
  OUTPUT_URI: gs://YOUR_BUCKET/output
  HF_HOME: /workspace/.cache/huggingface
""",
        encoding="utf-8",
    )
    template.write_text("name: ${display_name}\nimage: ${image_uri}\nconfig: ${env_CONFIG_URI}\n", encoding="utf-8")

    rendered = render_vertex_job(config, template)

    assert "name: demo-job" in rendered
    assert "image: REGION-docker.pkg.dev/PROJECT/REPOSITORY/image:latest" in rendered
    assert "config: gs://YOUR_BUCKET/config.yaml" in rendered

