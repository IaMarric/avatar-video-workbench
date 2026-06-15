# Avatar Video Workbench

[![CI](https://github.com/IaMarric/avatar-video-workbench/actions/workflows/ci.yml/badge.svg)](https://github.com/IaMarric/avatar-video-workbench/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Avatar Video Workbench is a small control plane for reproducible avatar media
experiments:

```text
authorized source photo -> Nano Banana 2 variants -> LTX LoRA dataset -> LTX LoRA training -> LTX 2.3 video
```

The project is built for fictional or properly authorized characters. It does
not include private datasets, generated identity assets, API keys, cloud
projects, or provider-specific secrets.

## What It Does

- Validates LoRA image datasets with paired captions.
- Turns one authorized source photo into ten themed training variants with
  Nano Banana 2 (`gemini-3.1-flash-image`), using a public-safe pirate preset by
  default.
- Writes an LTX trainer dataset and can submit an official LTX LoRA training
  CustomJob to Vertex AI.
- Creates a neutral project layout for avatar experiments.
- Compiles a complete experiment package with validation report, benchmark
  prompts, dataset manifest, run manifest, and optional Vertex AI job YAML.
- Produces optional local review artifacts: a dataset contact sheet and a
  motion storyboard MP4 from a source still.
- Renders Vertex AI CustomJob specs from safe templates.
- Stages and submits a real LTX image-to-video CustomJob on Vertex AI.
- Exports public-safe backend metadata for comparing LTX runs with ComfyUI or
  other workflows.
- Writes sanitized Vertex run reports from CustomJob metadata, optional logs,
  and backend output metadata.
- Runs a synthetic smoke demo outside the repository to prove the pipeline.
- Scans a folder before publication for credentials, absolute local paths,
  private cloud references, generated media, and model artifacts.
- Lets LTX 2.3 image-to-video jobs load a trained LoRA from a private GCS URI.

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
scripts/smoke-demo.sh
```

Generate a themed LoRA dataset from one authorized photo:

```bash
avw generate-character-dataset \
  --source-image "$AUTHORIZED_SOURCE_PHOTO" \
  --out-dir runs/pirate-avatar \
  --trigger "avwpirate person" \
  --theme pirate \
  --variant-count 10 \
  --vertex-project "$GOOGLE_CLOUD_PROJECT" \
  --vertex-location "$GOOGLE_CLOUD_LOCATION" \
  --auth-mode gcloud
```

Train an LTX LoRA from that dataset:

```bash
avw submit-ltx-lora-train \
  --run-id "$AVW_RUN_ID" \
  --gcs-root "$PRIVATE_GCS_RUN_PREFIX" \
  --dataset-dir runs/pirate-avatar \
  --trigger "avwpirate person" \
  --model-uri "hf://Lightricks/LTX-2/ltx-2-19b-dev.safetensors" \
  --text-encoder-uri "hf://Lightricks/LTX-2?include=text_encoder/**&include=tokenizer/**" \
  --region "$AVW_REGION" \
  --container-image "$AVW_CONTAINER_IMAGE"
```

For gated or rate-limited Hugging Face assets, store the token in Secret
Manager and add `--hf-token-secret "$HF_TOKEN_SECRET_NAME"`.

Submit a real LTX image-to-video job to Vertex AI:

```bash
avw submit-ltx-i2v \
  --run-id "$AVW_RUN_ID" \
  --gcs-root "$AVW_GCS_ROOT" \
  --input-image "$AVW_INPUT_IMAGE" \
  --prompt "$AVW_PROMPT" \
  --region "$AVW_REGION" \
  --container-image "$AVW_CONTAINER_IMAGE" \
  --lora-weights-uri "$TRAINED_LORA_GCS_URI"
```

The command stages the runner, config, and input image to GCS, then creates a
Vertex CustomJob. Outputs are written to the run prefix under GCS.

Export backend metadata for comparison:

```bash
avw export-backend-metadata \
  --input runs/demo-avatar/config/ltx_i2v.yaml \
  --out runs/demo-avatar/reports/backend-metadata.json
```

The export keeps prompt, model, dimensions, seed, sampling, and LoRA scale
metadata while omitting generated media, local paths, and cloud object URIs.

Write a sanitized Vertex run report:

```bash
avw vertex-run-report \
  --job-name "$VERTEX_CUSTOM_JOB_NAME" \
  --region "$AVW_REGION" \
  --out runs/demo-avatar/reports/vertex-run-report.json
```

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

## Project Links

- [Public demo](docs/demo.md): synthetic smoke demo and safe sharing rules.
- [Hugging Face report checker](https://huggingface.co/spaces/Iamarric/avatar-video-workbench-report-checker):
  public validator for sanitized Vertex reports and dataset manifests.
- [Community](docs/community.md): issues, discussions, and useful feedback.
- [Changelog](CHANGELOG.md): release history and notable changes.
- [Architecture](docs/architecture.md): repository boundary and pipeline
  diagram.
- [Roadmap](docs/roadmap.md): technical roadmap for reproducibility, safety,
  and backend support.
- [Backend metadata](docs/backend-metadata.md): public-safe comparison reports
  for LTX and ComfyUI-style workflows.
- [Vertex run reports](docs/vertex-run-reports.md): sanitized CustomJob,
  output, and log summaries.
- [Pre-staged model assets](docs/model-assets.md): private checkpoint and text
  encoder layouts for repeatable Vertex runs.
- [Contributing](CONTRIBUTING.md): setup, quality checks, and safety rules.

## Scope

This repo is intentionally the experiment control plane, not a bundled model
zoo. Large models, LoRA checkpoints, generated videos, source datasets, and
provider credentials stay outside git.

Local smoke demos and Vertex runs write generated media only under the output
directory or GCS prefix you choose. Do not commit those outputs.

See [docs/pirate-character-flow.md](docs/pirate-character-flow.md) for the
complete photo-to-LoRA-to-video route.
See [docs/demo.md](docs/demo.md) for a public-safe synthetic demo.

## Suggested GitHub Topics

`avatar-generation`, `lora`, `image-to-video`, `video-generation`, `vertex-ai`,
`comfyui`, `mlops`, `generative-ai`
