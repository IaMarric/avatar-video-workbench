from __future__ import annotations

from pathlib import Path

from avatar_video_workbench.vertex import preflight_vertex_job, render_vertex_job


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
  image_uri: region-docker.pkg.dev/project-id/repository/image:latest
  command: python demo.py
env:
  CONFIG_URI: https://storage.googleapis.com/bucket/config.yaml
  INPUT_IMAGE_URI: https://storage.googleapis.com/bucket/input.png
  OUTPUT_URI: https://storage.googleapis.com/bucket/output
  HF_HOME: /workspace/.cache/huggingface
""",
        encoding="utf-8",
    )
    template.write_text("name: ${display_name}\nimage: ${image_uri}\nconfig: ${env_CONFIG_URI}\n", encoding="utf-8")

    rendered = render_vertex_job(config, template)

    assert "name: demo-job" in rendered
    assert "image: region-docker.pkg.dev/project-id/repository/image:latest" in rendered
    assert "config: https://storage.googleapis.com/bucket/config.yaml" in rendered


def test_preflight_vertex_job_rejects_placeholders(tmp_path: Path) -> None:
    job_yaml = tmp_path / "job.yaml"
    job_yaml.write_text(
        """
displayName: demo-job
jobSpec:
  workerPoolSpecs:
    - machineSpec:
        machineType: a3-highgpu-1g
        acceleratorType: NVIDIA_H100_80GB
        acceleratorCount: 1
      containerSpec:
        imageUri: REGION-docker.pkg.dev/PROJECT/REPOSITORY/image:latest
        command:
          - python
          - run.py
""",
        encoding="utf-8",
    )

    result = preflight_vertex_job(job_yaml)

    assert result["ok"] is False
    assert any(item["code"] == "unresolved_placeholder" for item in result["errors"])


def test_preflight_vertex_job_accepts_complete_job(tmp_path: Path) -> None:
    job_yaml = tmp_path / "job.yaml"
    job_yaml.write_text(
        """
displayName: demo-job
jobSpec:
  workerPoolSpecs:
    - machineSpec:
        machineType: a3-highgpu-1g
        acceleratorType: NVIDIA_H100_80GB
        acceleratorCount: 1
      containerSpec:
        imageUri: us-central1-docker.pkg.dev/demo-project/runtime/image:latest
        command:
          - python
          - run.py
""",
        encoding="utf-8",
    )

    result = preflight_vertex_job(job_yaml)

    assert result["ok"] is True
