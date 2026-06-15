---
title: Avatar Video Workbench Report Checker
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# Hugging Face Space

This folder contains the public Hugging Face Space used to validate Avatar Video
Workbench reports before sharing them.

Live Space:
<https://huggingface.co/spaces/Iamarric/avatar-video-workbench-report-checker>

The app checks:

- sanitized Vertex run reports produced by `avw vertex-run-report`;
- dataset manifests produced by `avw compile-run` or LTX trainer manifests;
- accidental publication of cloud URIs, project resources, local paths, service
  accounts, API tokens, generated media references, and model artifacts.

It does not upload or process source photos, generated frames, videos, LoRA
weights, checkpoints, API keys, or private cloud resources.

## Local Run

```bash
python3 -m venv .venv-space
.venv-space/bin/pip install -r integrations/huggingface-space/requirements.txt
.venv-space/bin/python integrations/huggingface-space/app.py
```

## Space Files

Deploy the files in this directory to a public Gradio Space. Keep the Space on
CPU hardware; the validator is text-only.
