# Avatar Video Workbench

Avatar Video Workbench is a small control plane for reproducible avatar media
experiments:

```text
character brief -> curated image dataset -> benchmark package -> Vertex LTX image-to-video job -> output audit
```

The project is built for fictional or properly authorized characters. It does
not include private datasets, generated identity assets, API keys, cloud
projects, or provider-specific secrets.

## What It Does

- Validates LoRA image datasets with paired captions.
- Creates a neutral project layout for avatar experiments.
- Compiles a complete experiment package with validation report, benchmark
  prompts, dataset manifest, run manifest, and optional Vertex AI job YAML.
- Produces optional local review artifacts: a dataset contact sheet and a
  motion storyboard MP4 from a source still.
- Renders Vertex AI CustomJob specs from safe templates.
- Stages and submits a real LTX image-to-video CustomJob on Vertex AI.
- Runs a synthetic smoke demo outside the repository to prove the pipeline.
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

Run a local smoke test:

```bash
avw smoke-demo --out-dir /tmp/avatar-video-workbench-smoke --force
avw preflight-vertex --job-yaml /tmp/avatar-video-workbench-smoke/compiled/jobs/vertex-custom-job.yaml
```

Submit a real LTX image-to-video job to Vertex AI:

```bash
avw submit-ltx-i2v \
  --run-id "$AVW_RUN_ID" \
  --gcs-root "$AVW_GCS_ROOT" \
  --input-image "$AVW_INPUT_IMAGE" \
  --prompt "$AVW_PROMPT" \
  --region "$AVW_REGION" \
  --container-image "$AVW_CONTAINER_IMAGE"
```

The command stages the runner, config, and input image to GCS, then creates a
Vertex CustomJob. Outputs are written to the run prefix under GCS.

Run the publication safety scan:

```bash
avw scan-publication .
```

Compile a real local experiment package:

```bash
avw compile-run \
  --project-config runs/demo-avatar/avatar_project.yaml \
  --out-dir runs/demo-avatar/compiled \
  --with-previews
```

## Repository Layout

```text
configs/       Local project config examples
docs/          Pipeline, safety, and release notes
src/           CLI and library code
templates/     Vertex AI job templates
tests/         Unit tests
```

## Scope

This repo is intentionally the experiment control plane, not a bundled model
zoo. Large models, LoRA checkpoints, generated videos, source datasets, and
provider credentials stay outside git.

Local smoke demos and Vertex runs write generated media only under the output
directory or GCS prefix you choose. Do not commit those outputs.

## Suggested GitHub Topics

`avatar-generation`, `lora`, `image-to-video`, `video-generation`, `vertex-ai`,
`comfyui`, `mlops`, `generative-ai`
