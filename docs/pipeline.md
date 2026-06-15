# Pipeline

Avatar Video Workbench treats the repo as a reproducible control plane.

## 1. Character Definition

Start with a short character brief:

- fictional or properly authorized identity;
- stable trigger token;
- stable face/body descriptors;
- allowed wardrobe and settings;
- forbidden traits that cause identity drift;
- release and consent notes when a real person is involved.

Do not commit private source images. Commit only templates, examples, and
metadata that are safe to publish.

## 2. Dataset Curation

Use paired image/caption files:

```text
dataset/images/001_portrait.png
dataset/images/001_portrait.txt
```

Captions should separate:

- identity token;
- mutable clothing;
- camera/framing;
- environment;
- lighting;
- pose/action.

Run:

```bash
avw validate-dataset --images-dir dataset/images --trigger "demoavatar person"
```

Compile the experiment package:

```bash
avw compile-run \
  --project-config avatar_project.yaml \
  --out-dir compiled
```

This writes:

- `compiled/manifest.json`;
- `compiled/reports/dataset-validation.json`;
- `compiled/prompts/benchmark-prompts.yaml`;
- `compiled/jobs/vertex-custom-job.yaml` when a Vertex config/template is provided.

## 3. Image LoRA

Train the first LoRA on still images before video. Keep output in cloud storage
or local ignored directories. Store only sanitized configs and reports.

Recommended first target:

- 20-60 curated images;
- balanced close, half-body, full-body, indoor, outdoor;
- one stable trigger token;
- LoRA rank 16-64 depending on model and dataset;
- public-demo benchmark prompts before moving to video.

## 4. Still Benchmarks

Generate a fixed still grid:

- portrait;
- full body;
- walking;
- seated;
- low-light phone shot;
- bright outdoor phone shot;
- neutral background;
- motion-friendly hero still.

Pick 1-3 hero stills for video comparison.

## 5. Image-To-Video Comparison

Compare models with the same hero still and prompt:

- LTX-style I2V;
- Wan-style I2V;
- Hunyuan-style I2V;
- ComfyUI workflow route if useful.

Record:

- identity retention;
- face stability;
- body/hands;
- environment stability;
- motion naturalness;
- runtime and cost;
- failure notes.

## 6. Production Rule

A workflow is not production-ready until it can be rerun from tracked config,
not only from a manually clicked UI graph.

Use `avw smoke-demo --out-dir /tmp/avatar-video-workbench-smoke --force` after
installation to verify the local pipeline without committing generated assets.
