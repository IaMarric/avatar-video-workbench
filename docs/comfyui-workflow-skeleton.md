# Public-Safe ComfyUI Workflow Skeleton

This example documents the shape of a ComfyUI comparison workflow without
including generated media, private model assets, bucket names, or machine-local
paths. Treat every uppercase value as a user-provided placeholder.

## `backend-metadata.json` Mapping

| Metadata field | ComfyUI setting | User provides |
| --- | --- | --- |
| `run_id` | `AVW_RunMetadata.run_id` | stable experiment name |
| `input_image_ref` | `AVW_LoadImage.image_ref` | authorized source image reference |
| `prompt` | `AVW_PositivePrompt.text` | motion/video prompt |
| `negative_prompt` | `AVW_NegativePrompt.text` | safety and quality exclusions |
| `trigger_token` | prefix in positive prompt | trained identity trigger |
| `base_model` | `AVW_LoadCheckpoint.model_name` | placeholder checkpoint name |
| `lora_name` | `AVW_LoadLoRA.lora_name` | placeholder LoRA adapter name |
| `lora_strength` | `AVW_LoadLoRA.strength_model` | adapter strength |
| `seed` | `AVW_Sampler.seed` | reproducible integer seed |
| `steps` | `AVW_Sampler.steps` | sampling step count |
| `cfg_scale` | `AVW_Sampler.cfg` | guidance scale |
| `width` / `height` | `AVW_EmptyLatentImage` size | target frame size |
| `fps` / `frame_count` | `AVW_VideoCombine` settings | video timing |

## Skeleton

```json
{
  "workflow_note": "Public skeleton only; replace placeholders locally.",
  "nodes": {
    "AVW_RunMetadata": { "run_id": "PLACEHOLDER_RUN_ID" },
    "AVW_LoadImage": { "image_ref": "USER_INPUT_IMAGE_REFERENCE" },
    "AVW_LoadCheckpoint": { "model_name": "PLACEHOLDER_BASE_MODEL" },
    "AVW_LoadLoRA": {
      "lora_name": "PLACEHOLDER_LORA_ADAPTER",
      "strength_model": "USER_LORA_STRENGTH"
    },
    "AVW_PositivePrompt": { "text": "TRIGGER_TOKEN + USER_VIDEO_PROMPT" },
    "AVW_NegativePrompt": { "text": "USER_NEGATIVE_PROMPT" },
    "AVW_EmptyLatentImage": { "width": "USER_WIDTH", "height": "USER_HEIGHT" },
    "AVW_Sampler": { "seed": "USER_SEED", "steps": "USER_STEPS", "cfg": "USER_CFG_SCALE" },
    "AVW_VideoCombine": { "fps": "USER_FPS", "frame_count": "USER_FRAME_COUNT" }
  }
}
```

The `AVW_*` names are neutral placeholders, not required custom nodes. If a
real workflow uses optional community nodes, document them separately and keep
that workflow outside git until all model names, media references, and output
locations are sanitized.
