# Codex For OSS Notes

Avatar Video Workbench is a public control plane for avatar video experiments.
It turns scattered local scripts, prompt files, cloud job specs, and model
comparison notes into a reproducible open source workflow.

## Why Codex Helps

- Convert exploratory ComfyUI and cloud scripts into maintained Python modules.
- Keep safety checks close to the code path that prepares a public release.
- Generate and review provider-specific job templates without committing
  private buckets, keys, generated media, or model artifacts.
- Maintain tests for dataset validation, release scans, and Vertex AI job
  rendering.
- Expand the project with adapters for more image-to-video backends while
  keeping the public interface stable.

## Current OSS Surface

- `avw init` creates a clean experiment scaffold.
- `avw validate-dataset` validates paired image and caption datasets.
- `avw compile-run` writes a validated experiment manifest, reports, and
  benchmark prompt matrix, with optional review previews.
- `avw smoke-demo` proves the pipeline locally with synthetic temporary assets,
  a contact sheet, and a motion storyboard video.
- `avw submit-ltx-i2v` stages a real LTX image-to-video run and submits it as a
  Vertex AI CustomJob.
- `avw render-vertex-job` renders cloud job specs from explicit user configs.
- `avw preflight-vertex` validates rendered Vertex CustomJob YAML before launch.
- `avw scan-publication` blocks credentials, private paths, generated media,
  and checkpoint files before publication.

## Near-Term Roadmap

- Add GitHub Actions for tests and publication scanning.
- Add a run registry and benchmark manifest.
- Add cloud cost and runtime report parsing.
- Add adapters for LTX, Wan, Hunyuan, and ComfyUI workflow metadata.
- Publish a minimal demo that uses synthetic temporary assets only.

## Public Data Policy

The repository intentionally ships without generated images, generated videos,
source datasets, private identity assets, LoRA checkpoints, cloud buckets, or
service account files.
