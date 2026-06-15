# Contributing

Avatar Video Workbench is an open source control plane for reproducible avatar
LoRA and image-to-video experiments. Contributions should improve reproducible
workflows, cloud job safety, provider adapters, documentation, or publication
checks.

## Ground Rules

- Do not commit private source photos, generated identity images, generated
  videos, LoRA weights, model checkpoints, API keys, cloud bucket names,
  service account files, or local absolute paths.
- Do not add NSFW, erotic, explicit, or non-consensual identity workflows.
- Use only synthetic assets, licensed assets, or clearly authorized source
  photos in examples.
- Keep examples runnable without private credentials whenever possible.
- Prefer clear failures over hidden fallbacks when a cloud/provider integration
  is not configured.

## Local Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[test]"
.venv/bin/pytest
.venv/bin/avw scan-publication .
```

## Useful First Tasks

- Add provider adapters behind the existing CLI shape.
- Improve reports that parse runtime, cost, and output quality.
- Add safer dataset validation rules.
- Improve documentation for cloud setup and no-private-asset demos.
- Add tests for new job renderers or runners.

## Pull Request Checklist

- Tests pass locally with `.venv/bin/pytest`.
- Publication scan returns `[]`.
- New cloud paths keep credentials and generated media outside git.
- Docs are updated when CLI behavior changes.
- New examples use synthetic or authorized inputs only.

## Communication

Open a GitHub issue before large changes. Small documentation fixes can go
straight to a pull request.
