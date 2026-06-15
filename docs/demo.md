# Public Demo

This demo is safe to share because it uses synthetic local assets and keeps all
generated media outside git.

## Local Smoke Demo

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[test]"
scripts/smoke-demo.sh
```

Expected result:

- the smoke demo writes a synthetic source image, manifest, contact sheet, and
  storyboard under `/tmp/avatar-video-workbench-smoke`;
- Vertex preflight validates the rendered CustomJob YAML;
- publication scan returns `[]`.

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
