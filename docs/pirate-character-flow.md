# Pirate Character Flow

This flow turns one authorized source photo into a themed character video
pipeline:

```text
source photo -> Nano Banana 2 variants -> LTX LoRA dataset -> LTX LoRA training -> LTX 2.3 I2V with LoRA
```

The default theme is a family-friendly pirate transformation. The repo does not
ship source photos, generated variants, LoRA weights, or videos.

## 1. Generate the Dataset

Nano Banana 2 is the Gemini image model `gemini-3.1-flash-image`. The command
uses image editing: it sends the source image plus ten edit prompts by default.

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

Outputs:

- `source/source.*`: normalized private source image copy.
- `dataset/images/*.png`: generated training variants.
- `dataset/images/*.txt`: paired captions containing the trigger token.
- `dataset.json`: dataset metadata for the official LTX trainer.
- `ltx_trainer/dataset.json`: legacy copy for older manual workflows.
- `reports/nano-banana-requests.jsonl`: exact prompts used for generation.
- `reports/dataset-contact-sheet.png`: local review sheet.
- `manifest.json`: run manifest.

Use `--plan-only` to write prompts and metadata without calling Gemini.
Use the default `--auth-mode sdk` when Application Default Credentials or a
Gemini API key are configured. Use `--auth-mode gcloud` when the active
`gcloud` account can call Vertex AI but local ADC is not available.

## 2. Train an LTX LoRA

The official LTX trainer requires local model assets inside the training job:

- an LTX 2.3 `.safetensors` checkpoint;
- a Gemma text encoder directory;
- Linux/CUDA with enough VRAM. H100-class hardware is the recommended route.

Use the public Hugging Face model assets directly, or stage equivalent files in
a private GCS prefix:

```bash
avw submit-ltx-lora-train \
  --run-id "$AVW_RUN_ID" \
  --gcs-root "$PRIVATE_GCS_RUN_PREFIX" \
  --dataset-dir runs/pirate-avatar \
  --trigger "avwpirate person" \
  --model-uri "hf://Lightricks/LTX-2/ltx-2-19b-dev.safetensors" \
  --text-encoder-uri "hf://Lightricks/LTX-2?include=text_encoder/**&include=tokenizer/**" \
  --region "$VERTEX_REGION" \
  --container-image "$GPU_CONTAINER_IMAGE" \
  --machine-type a3-highgpu-1g \
  --accelerator-type NVIDIA_H100_80GB \
  --accelerator-count 1
```

The job clones the official LTX-2 trainer, preprocesses
`dataset.json` with a `512x512x1` image bucket, trains a LoRA, and
uploads checkpoints to the run output prefix.

`--model-uri` and `--text-encoder-uri` accept either `hf://...` or `gs://...`.
For `hf://` directories, repeated `include` query parameters limit which repo
paths are downloaded. When a Hugging Face asset is gated, store the token in
Google Secret Manager and pass the secret name with `--hf-token-secret`.
For repeated Vertex runs, see [model-assets.md](model-assets.md) for the
pre-staged checkpoint and text encoder layout.

## 3. Generate Video With the LoRA

Use one selected generated image as the first frame and pass the trained LoRA
checkpoint URI into the LTX 2.3 image-to-video job:

```bash
avw submit-ltx-i2v \
  --run-id "$AVW_VIDEO_RUN_ID" \
  --gcs-root "$PRIVATE_GCS_RUN_PREFIX" \
  --input-image runs/pirate-avatar/dataset/images/001.png \
  --prompt "avwpirate person stands on a wooden ship deck at golden hour, subtle confident smile, sea wind moving clothing, slow handheld push-in, stable identity, no text." \
  --negative-prompt "low quality, distorted face, identity drift, text, watermark" \
  --region "$VERTEX_REGION" \
  --container-image "$GPU_CONTAINER_IMAGE" \
  --machine-type "$VIDEO_MACHINE_TYPE" \
  --accelerator-type "$VIDEO_ACCELERATOR_TYPE" \
  --accelerator-count 1 \
  --lora-weights-uri "$TRAINED_LORA_GCS_URI" \
  --lora-scale 1.0
```

The output is an MP4 plus a manifest. Generated outputs stay in the private run
directory or GCS prefix.

## Publication Rules

- Use only your own photo, a licensed source photo, or a clearly authorized
  person.
- Do not commit `runs/`, generated images, videos, LoRA weights, model
  checkpoints, API keys, GCS prefixes, or project IDs.
- Run `avw scan-publication .` before pushing public changes.
