# Pre-Staged Model Assets

Pre-staging model assets avoids downloading the same large files on every
Vertex CustomJob. Keep the assets in a private bucket or private model cache;
commit only the layout and placeholder variables.

## LTX Checkpoint

`submit-ltx-lora-train` expects `--model-uri` to point to one LTX checkpoint
file. Use either a Hugging Face file URI or a private staged object URI held in
an environment variable:

```bash
export LTX_MODEL_URI="$PRIVATE_LTX_MODEL_CHECKPOINT_URI"
```

Recommended private layout:

```text
$MODEL_ASSET_PREFIX/
  ltx-2/
    checkpoints/
      ltx-2-19b-dev.safetensors
    checksums/
      SHA256SUMS.txt
```

Pass the checkpoint object:

```bash
avw submit-ltx-lora-train \
  --model-uri "$LTX_MODEL_URI" \
  --text-encoder-uri "$LTX_TEXT_ENCODER_URI" \
  ...
```

## Text Encoder And Tokenizer

`--text-encoder-uri` should point to a directory-like prefix containing the text
encoder and tokenizer files required by the trainer. For Hugging Face, the repo
query can include only those folders. For staged private assets, keep the same
relative shape:

```text
$MODEL_ASSET_PREFIX/
  ltx-2/
    text_encoder/
      config.json
      generation_config.json
      model.safetensors.index.json
      model-00001-of-000NN.safetensors
      model-00002-of-000NN.safetensors
    tokenizer/
      special_tokens_map.json
      tokenizer.json
      tokenizer_config.json
```

Set a placeholder variable for the private prefix:

```bash
export LTX_TEXT_ENCODER_URI="$PRIVATE_LTX_TEXT_ENCODER_PREFIX"
```

The trainer copies staged assets into the job workspace before preprocessing
and training. The final local layout inside the container is:

```text
/workspace/.../
  models/
    ltx-2.3.safetensors
    text_encoder/
      text_encoder/
      tokenizer/
```

## Staging Checklist

Before launching a GPU job, verify:

- the checkpoint URI resolves to exactly one `.safetensors` file;
- the text encoder URI contains both `text_encoder/` and `tokenizer/`;
- all sharded files listed by `model.safetensors.index.json` are present;
- the Vertex service account can read the private model asset prefix;
- checksums are stored outside the model directory or in a `checksums/` folder;
- the model asset prefix is not inside this repository.

## Failure Modes

- Missing checkpoint: the job fails before trainer startup when the model file
  cannot be copied into the workspace.
- Wrong text encoder root: tokenizer or text encoder loading fails because the
  expected child folders are absent.
- Partial upload: a safetensors index references shard files that were not
  staged.
- Permission mismatch: the job starts, but asset download returns an
  authorization error.
- Gated Hugging Face asset without token: download fails before training; use
  `--hf-token-secret` for a Secret Manager token reference.
- Stale cache: the job runs with an older model than expected; compare the
  checksum manifest before launching long training runs.

## Publication Boundary

Do not commit:

- bucket names, object URIs, or project IDs;
- local model cache paths;
- downloaded model weights;
- LoRA checkpoints;
- generated videos or image datasets.

Use `avw scan-publication .` before publishing docs or config examples.
