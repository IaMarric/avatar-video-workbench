# Vertex LTX I2V

`avw submit-ltx-i2v` runs a real Vertex AI CustomJob for LTX image-to-video.
It stages three private inputs to GCS:

- the Vertex runner script;
- a generated LTX config;
- the source image.

The job downloads those inputs, loads the LTX Diffusers pipeline on GPU,
generates an MP4, writes a manifest, writes a contact sheet when `ffmpeg` is
available in the container, then uploads outputs back to GCS.

## Requirements

- `gcloud` authenticated to a project with Vertex AI enabled.
- `gsutil` access to the target GCS prefix.
- A GPU-capable Vertex container image with CUDA and Python. The runner installs
  missing Python packages at startup, including Diffusers, Transformers, and
  PEFT for LoRA loading.
- Vertex Custom Training GPU quota in the target region.
- A source image that you have rights to use.

## Run

Set real values in your shell:

```bash
export AVW_RUN_ID="avatar-ltx-$(date -u +%Y%m%d-%H%M%S)"
export AVW_GCS_ROOT="$PRIVATE_GCS_RUN_PREFIX"
export AVW_INPUT_IMAGE="/path/to/source-image.png"
export AVW_REGION="asia-southeast1"
export AVW_CONTAINER_IMAGE="region-docker.pkg.dev/project/repository/gpu-runtime:tag"
export AVW_PROMPT="A handheld phone video of the same fictional avatar making a subtle natural head movement."
export AVW_LORA_URI="$TRAINED_LTX_LORA_GCS_URI"
```

Submit:

```bash
avw submit-ltx-i2v \
  --run-id "$AVW_RUN_ID" \
  --gcs-root "$AVW_GCS_ROOT" \
  --input-image "$AVW_INPUT_IMAGE" \
  --prompt "$AVW_PROMPT" \
  --region "$AVW_REGION" \
  --container-image "$AVW_CONTAINER_IMAGE" \
  --lora-weights-uri "$AVW_LORA_URI"
```

For a fast infrastructure smoke, keep the defaults: `256x256`, `17` frames and
`4` inference steps. Increase resolution, frames and steps only after the
pipeline succeeds end to end.

Omit `--lora-weights-uri` for a base-model I2V run. Include it after
`submit-ltx-lora-train` has produced a LoRA checkpoint in a private GCS prefix.

## Outputs

The GCS output prefix contains:

- `video/*.mp4`;
- `reports/*_manifest.json`;
- `reports/*_contact_sheet.jpg` when the container includes `ffmpeg`;
- copied input/config files for audit.

Generated outputs stay outside git.

Use `avw vertex-run-report` after a CustomJob finishes to produce a sanitized
review report from the Vertex job metadata. Use `avw export-backend-metadata`
on the runtime manifest when you also want model, prompt, sampling, and LoRA
settings attached to that report.
