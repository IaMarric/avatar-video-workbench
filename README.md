# Avatar Video Workbench

Avatar Video Workbench is a small control plane for reproducible avatar media
experiments:

```text
character brief -> curated image dataset -> LoRA training -> still benchmarks -> image-to-video comparison
```

The project is built for fictional or properly authorized characters. It does
not include private datasets, generated identity assets, API keys, cloud
projects, or provider-specific secrets.

## What It Does

- Validates LoRA image datasets with paired captions.
- Creates a neutral project layout for avatar experiments.
- Renders Vertex AI CustomJob specs from safe templates.
- Scans a folder before publication for credentials, absolute local paths,
  private cloud references, generated media, and model artifacts.
- Provides example configs for Qwen-style image LoRA and LTX/Wan-style
  image-to-video pipelines.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[test]"
.venv/bin/pytest
```

## Quick Start

Create a local experiment scaffold:

```bash
avw init runs/demo-avatar --name demo-avatar --trigger "demoavatar person"
```

Validate a LoRA dataset:

```bash
avw validate-dataset \
  --images-dir runs/demo-avatar/dataset/images \
  --trigger "demoavatar person" \
  --out runs/demo-avatar/reports/dataset-validation.json
```

Render a Vertex CustomJob YAML from an example config:

```bash
avw render-vertex-job \
  --config configs/ltx_i2v.example.yaml \
  --template templates/vertex_custom_job.yaml \
  --out runs/demo-avatar/jobs/ltx-i2v-custom-job.yaml
```

Run the publication safety scan:

```bash
avw scan-publication .
```

## Repository Layout

```text
configs/       Example experiment configs with placeholders only
docs/          Pipeline, safety, and release notes
src/           CLI and library code
templates/     Vertex AI job templates
tests/         Unit tests
```

## Scope

This repo is intentionally the control plane, not a bundled model zoo. Large
models, LoRA checkpoints, generated videos, source datasets, and provider
credentials stay outside git.

## Suggested GitHub Topics

`avatar-generation`, `lora`, `image-to-video`, `video-generation`, `vertex-ai`,
`comfyui`, `mlops`, `generative-ai`
