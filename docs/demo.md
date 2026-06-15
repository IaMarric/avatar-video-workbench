# Public Demo

This demo is safe to share because it uses synthetic local assets and keeps all
generated media outside git.

## Local Smoke Demo

Run the reproducible terminal demo from a clean checkout:

```bash
scripts/synthetic_smoke_demo.sh
```

The script prepares `.venv` if needed, then runs the same workflow manually shown
below. Generated media is written only under `/tmp/avatar-video-workbench-smoke`.

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[test]"
scripts/smoke-demo.sh
```

Expected output:

- `avw smoke-demo` prints a JSON manifest with generated paths under
  `/tmp/avatar-video-workbench-smoke` (or the platform-resolved temp path) and
  writes synthetic demo images, captions, a contact sheet, and a storyboard there;
- `avw preflight-vertex` prints JSON containing `"ok": true` and `"errors": []`
  for the generated `compiled/jobs/vertex-custom-job.yaml`;
- `avw scan-publication .` prints exactly `[]`.

Do not copy files from `/tmp/avatar-video-workbench-smoke` into git unless they
are plain configuration or documentation assets that pass the publication scan.

## Cloud Demo Shape

The cloud path uses private GCS prefixes and Vertex AI CustomJobs:

```text
authorized source photo
  -> Nano Banana 2 variants
  -> LTX trainer dataset
  -> LTX LoRA training CustomJob
  -> LTX 2.3 image-to-video CustomJob with the trained LoRA
  -> MP4 and manifest in private GCS output
```

The public repository documents and submits that route, but it does not include
the generated images, generated videos, LoRA checkpoints, private source
photos, cloud buckets, project IDs, or credentials.

## Safe Sharing Assets

Use these in public posts:

- repository URL;
- architecture description;
- command snippets;
- CI link;
- sanitized logs that remove bucket names, project IDs, account names, and
  source-image references.

Avoid these in public posts:

- private source photos;
- generated faces or identity assets;
- generated MP4s unless they use a synthetic or explicitly public character;
- cloud bucket names, project IDs, service account names, or tokens.
