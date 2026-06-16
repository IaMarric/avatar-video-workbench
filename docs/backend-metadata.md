# Backend Metadata Export

`avw export-backend-metadata` writes a small JSON report for comparing Avatar
Video Workbench runs against ComfyUI workflows or other image-to-video
backends.

The export keeps comparable settings and removes private execution details:

- backend and runtime name;
- model ID;
- positive and negative prompts;
- width, height, frame count, FPS, and duration;
- seed, steps, and guidance scale when available;
- LoRA enabled/scale state without checkpoint paths or cloud URIs;
- runtime metrics when they already exist in the AVW manifest.

## Supported Inputs

The command accepts:

- an LTX I2V config such as `runs/example/config/ltx_i2v.yaml`;
- an LTX I2V runtime manifest written by the Vertex runner;
- a `submission-manifest.json` when the sibling `config/ltx_i2v.yaml` is still
  available in the staging directory.

Submission manifests alone do not contain prompt and model settings. If the
staged config is missing, the command fails instead of guessing.

## Usage

```bash
avw export-backend-metadata \
  --input runs/example/config/ltx_i2v.yaml \
  --out runs/example/reports/backend-metadata.json
```

For a Vertex runtime result:

```bash
avw export-backend-metadata \
  --input runs/example/output/reports/example_manifest.json \
  --out runs/example/reports/backend-metadata.json
```

See [examples/ltx-i2v-backend-metadata.json](examples/ltx-i2v-backend-metadata.json)
for the public-safe JSON shape. The contract is documented in
[`schemas/backend-metadata.schema.json`](../schemas/backend-metadata.schema.json).
