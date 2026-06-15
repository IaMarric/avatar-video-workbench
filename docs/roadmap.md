# Project Roadmap

Avatar Video Workbench is a control plane for reproducible avatar LoRA and
image-to-video experiments. The roadmap focuses on making runs easier to
repeat, inspect, and compare while keeping private assets out of git.

## Current Baseline

- Synthetic local smoke demo.
- Dataset validation for image/caption LoRA datasets.
- Publication scan for credentials, private paths, generated media, and model
  artifacts.
- Vertex AI CustomJob submission for LTX image-to-video.
- Vertex AI CustomJob submission for LTX LoRA training.
- LTX 2.3 image-to-video with a trained LoRA from a private GCS URI.
- CI that runs tests and publication scanning.

## Near Term

### Safer Public Examples

- Extend the repo-native architecture diagram when new backends are added.
- Keep the synthetic terminal demo aligned with CLI changes.
- Expand safe example configs that do not require private credentials.

### Run Evidence

- Parse Vertex job state, timing, machine type, accelerator type, and output
  metadata into sanitized run reports.
- Add a compact run summary format for comparing outputs across backends.
- Keep private bucket names, project IDs, account names, and generated media out
  of public examples.

### Dataset And Publication Safety

- Harden dataset validation before training.
- Improve trigger-token consistency checks.
- Make publication scan findings easier to fix from the CLI output.

### Cloud Runner Ergonomics

- Document pre-staged model asset layouts for faster cloud runs.
- Support clearer errors for missing model assets, unavailable GPUs, and failed
  LoRA loading.
- Keep cloud credentials and generated outputs external to the repository.

## Later

- Add backend metadata export for tools such as ComfyUI.
- Add additional cloud runner adapters behind the same run manifest shape.
- Add benchmark manifests for comparing model, seed, prompt, dimensions, and
  runtime.

## Non-Goals

- Bundling model weights, LoRA checkpoints, generated videos, or private source
  photos.
- Publishing real identity examples without explicit authorization.
- Adding NSFW, explicit, or non-consensual identity workflows.
