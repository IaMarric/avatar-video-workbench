from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .character_flow import CharacterDatasetOptions, create_character_dataset
from .cloud import LtxI2VSubmitOptions, LtxLoraTrainSubmitOptions, submit_ltx_i2v, submit_ltx_lora_train
from .config import WorkbenchError, write_json, write_yaml
from .datasets import DatasetValidationOptions, validate_dataset
from .experiments import CompileRunOptions, SmokeDemoOptions, compile_run, create_smoke_demo
from .publication import findings_as_dicts, format_findings, scan_publication
from .vertex import preflight_vertex_job, render_vertex_job


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="avw", description="Avatar Video Workbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create a local avatar experiment scaffold")
    init_parser.add_argument("path")
    init_parser.add_argument("--name", required=True)
    init_parser.add_argument("--trigger", required=True)
    init_parser.set_defaults(func=_cmd_init)

    validate_parser = subparsers.add_parser("validate-dataset", help="Validate paired LoRA images and captions")
    validate_parser.add_argument("--images-dir", required=True)
    validate_parser.add_argument("--trigger", required=True)
    validate_parser.add_argument("--out")
    validate_parser.add_argument("--min-images", type=int, default=1)
    validate_parser.add_argument("--allow-missing-trigger", action="store_true")
    validate_parser.set_defaults(func=_cmd_validate_dataset)

    render_parser = subparsers.add_parser("render-vertex-job", help="Render a Vertex CustomJob YAML from config")
    render_parser.add_argument("--config", required=True)
    render_parser.add_argument("--template", required=True)
    render_parser.add_argument("--out", required=True)
    render_parser.set_defaults(func=_cmd_render_vertex_job)

    compile_parser = subparsers.add_parser("compile-run", help="Compile a validated avatar experiment package")
    compile_parser.add_argument("--project-config", required=True)
    compile_parser.add_argument("--out-dir", required=True)
    compile_parser.add_argument("--vertex-config")
    compile_parser.add_argument("--vertex-template")
    compile_parser.add_argument("--with-previews", action="store_true")
    compile_parser.set_defaults(func=_cmd_compile_run)

    preflight_parser = subparsers.add_parser("preflight-vertex", help="Validate a rendered Vertex CustomJob YAML")
    preflight_parser.add_argument("--job-yaml", required=True)
    preflight_parser.add_argument("--json-out")
    preflight_parser.set_defaults(func=_cmd_preflight_vertex)

    smoke_parser = subparsers.add_parser("smoke-demo", help="Run an end-to-end demo with synthetic temp assets")
    smoke_parser.add_argument("--out-dir", required=True)
    smoke_parser.add_argument("--force", action="store_true")
    smoke_parser.set_defaults(func=_cmd_smoke_demo)

    character_parser = subparsers.add_parser(
        "generate-character-dataset",
        help="Turn one authorized photo into a themed LoRA image dataset with Nano Banana 2",
    )
    character_parser.add_argument("--source-image", required=True)
    character_parser.add_argument("--out-dir", required=True)
    character_parser.add_argument("--trigger", required=True)
    character_parser.add_argument("--theme", default="pirate")
    character_parser.add_argument("--variant-count", type=int, default=10)
    character_parser.add_argument("--model", default="gemini-3.1-flash-image")
    character_parser.add_argument("--vertex-project")
    character_parser.add_argument("--vertex-location")
    character_parser.add_argument("--auth-mode", choices=["sdk", "gcloud"], default="sdk")
    character_parser.add_argument("--plan-only", action="store_true")
    character_parser.add_argument("--ltx-model-path")
    character_parser.add_argument("--ltx-text-encoder-path")
    character_parser.add_argument("--training-steps", type=int, default=1200)
    character_parser.add_argument("--resolution-bucket", default="512x512x1")
    character_parser.set_defaults(func=_cmd_generate_character_dataset)

    ltx_parser = subparsers.add_parser("submit-ltx-i2v", help="Stage and submit a real LTX image-to-video Vertex job")
    ltx_parser.add_argument("--run-id", required=True)
    ltx_parser.add_argument("--gcs-root", required=True)
    ltx_parser.add_argument("--input-image", required=True)
    ltx_parser.add_argument("--prompt", required=True)
    ltx_parser.add_argument("--negative-prompt", default="cartoon, CGI, distorted face, extra fingers, text, watermark")
    ltx_parser.add_argument("--region", required=True)
    ltx_parser.add_argument("--container-image", required=True)
    ltx_parser.add_argument("--machine-type", default="a3-highgpu-1g")
    ltx_parser.add_argument("--accelerator-type", default="NVIDIA_H100_80GB")
    ltx_parser.add_argument("--accelerator-count", type=int, default=1)
    ltx_parser.add_argument("--boot-disk-type", default="pd-ssd")
    ltx_parser.add_argument("--boot-disk-size-gb", type=int, default=1000)
    ltx_parser.add_argument("--staging-dir", default="runs/vertex-staging")
    ltx_parser.add_argument("--model-id", default="dg845/LTX-2.3-Diffusers")
    ltx_parser.add_argument("--width", type=int, default=256)
    ltx_parser.add_argument("--height", type=int, default=256)
    ltx_parser.add_argument("--num-frames", type=int, default=17)
    ltx_parser.add_argument("--fps", type=int, default=8)
    ltx_parser.add_argument("--num-inference-steps", type=int, default=4)
    ltx_parser.add_argument("--guidance-scale", type=float, default=2.5)
    ltx_parser.add_argument("--seed", type=int, default=1234)
    ltx_parser.add_argument("--no-spot", action="store_true")
    ltx_parser.add_argument("--dry-run", action="store_true")
    ltx_parser.add_argument("--lora-weights-uri")
    ltx_parser.add_argument("--lora-scale", type=float, default=1.0)
    ltx_parser.set_defaults(func=_cmd_submit_ltx_i2v)

    lora_train_parser = subparsers.add_parser(
        "submit-ltx-lora-train",
        help="Stage a generated dataset and submit an LTX LoRA trainer CustomJob",
    )
    lora_train_parser.add_argument("--run-id", required=True)
    lora_train_parser.add_argument("--gcs-root", required=True)
    lora_train_parser.add_argument("--dataset-dir", required=True)
    lora_train_parser.add_argument("--trigger", required=True)
    lora_train_parser.add_argument("--model-uri", required=True)
    lora_train_parser.add_argument("--text-encoder-uri", required=True)
    lora_train_parser.add_argument("--region", required=True)
    lora_train_parser.add_argument("--container-image", required=True)
    lora_train_parser.add_argument("--machine-type", default="a3-highgpu-1g")
    lora_train_parser.add_argument("--accelerator-type", default="NVIDIA_H100_80GB")
    lora_train_parser.add_argument("--accelerator-count", type=int, default=1)
    lora_train_parser.add_argument("--boot-disk-type", default="pd-ssd")
    lora_train_parser.add_argument("--boot-disk-size-gb", type=int, default=1000)
    lora_train_parser.add_argument("--staging-dir", default="runs/vertex-staging")
    lora_train_parser.add_argument("--resolution-bucket", default="512x512x1")
    lora_train_parser.add_argument("--training-steps", type=int, default=1200)
    lora_train_parser.add_argument("--trainer-repo", default="https://github.com/Lightricks/LTX-2.git")
    lora_train_parser.add_argument("--trainer-ref", default="main")
    lora_train_parser.add_argument("--validation-prompt")
    lora_train_parser.add_argument(
        "--hf-token-secret",
        help="Secret Manager secret name or resource containing an HF token for gated Hugging Face assets",
    )
    lora_train_parser.add_argument("--no-spot", action="store_true")
    lora_train_parser.add_argument("--dry-run", action="store_true")
    lora_train_parser.set_defaults(func=_cmd_submit_ltx_lora_train)

    scan_parser = subparsers.add_parser("scan-publication", help="Scan a project before public release")
    scan_parser.add_argument("path")
    scan_parser.add_argument("--json-out")
    scan_parser.set_defaults(func=_cmd_scan_publication)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except WorkbenchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _cmd_init(args: argparse.Namespace) -> int:
    root = Path(args.path).expanduser().resolve()
    for subdir in [
        "dataset/images",
        "configs",
        "jobs",
        "reports",
        "outputs/stills",
        "outputs/videos",
    ]:
        (root / subdir).mkdir(parents=True, exist_ok=True)

    project = {
        "project": {
            "name": args.name,
            "trigger": args.trigger,
            "character_policy": "fictional_or_authorized_public_demo",
        },
        "dataset": {
            "images_dir": "dataset/images",
            "min_images": 12,
        },
        "pipeline": [
            "curate_dataset",
            "train_image_lora",
            "generate_still_benchmarks",
            "select_hero_stills",
            "run_image_to_video_comparison",
        ],
    }
    write_yaml(root / "avatar_project.yaml", project)
    (root / "dataset/images/.gitkeep").write_text("", encoding="utf-8")
    print(str(root))
    return 0


def _cmd_validate_dataset(args: argparse.Namespace) -> int:
    result = validate_dataset(
        DatasetValidationOptions(
            images_dir=Path(args.images_dir),
            trigger=args.trigger,
            min_images=args.min_images,
            require_trigger=not args.allow_missing_trigger,
        )
    )
    if args.out:
        write_json(Path(args.out).expanduser().resolve(), result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["ok"] else 1


def _cmd_render_vertex_job(args: argparse.Namespace) -> int:
    output = render_vertex_job(Path(args.config), Path(args.template))
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output, encoding="utf-8")
    print(str(out_path))
    return 0


def _cmd_compile_run(args: argparse.Namespace) -> int:
    manifest = compile_run(
        CompileRunOptions(
            project_config=Path(args.project_config),
            out_dir=Path(args.out_dir),
            vertex_config=Path(args.vertex_config) if args.vertex_config else None,
            vertex_template=Path(args.vertex_template) if args.vertex_template else None,
            with_previews=args.with_previews,
        )
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


def _cmd_preflight_vertex(args: argparse.Namespace) -> int:
    result = preflight_vertex_job(Path(args.job_yaml))
    if args.json_out:
        write_json(Path(args.json_out).expanduser().resolve(), result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["ok"] else 1


def _cmd_smoke_demo(args: argparse.Namespace) -> int:
    manifest = create_smoke_demo(SmokeDemoOptions(out_dir=Path(args.out_dir), force=args.force))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


def _cmd_generate_character_dataset(args: argparse.Namespace) -> int:
    manifest = create_character_dataset(
        CharacterDatasetOptions(
            source_image=Path(args.source_image),
            out_dir=Path(args.out_dir),
            trigger=args.trigger,
            theme=args.theme,
            variant_count=args.variant_count,
            model=args.model,
            vertex_project=args.vertex_project,
            vertex_location=args.vertex_location,
            auth_mode=args.auth_mode,
            plan_only=args.plan_only,
            ltx_model_path=args.ltx_model_path,
            ltx_text_encoder_path=args.ltx_text_encoder_path,
            training_steps=args.training_steps,
            resolution_bucket=args.resolution_bucket,
        )
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


def _cmd_submit_ltx_i2v(args: argparse.Namespace) -> int:
    result = submit_ltx_i2v(
        LtxI2VSubmitOptions(
            run_id=args.run_id,
            gcs_root=args.gcs_root,
            input_image=Path(args.input_image),
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            region=args.region,
            container_image=args.container_image,
            machine_type=args.machine_type,
            accelerator_type=args.accelerator_type,
            accelerator_count=args.accelerator_count,
            boot_disk_type=args.boot_disk_type,
            boot_disk_size_gb=args.boot_disk_size_gb,
            staging_dir=Path(args.staging_dir),
            model_id=args.model_id,
            width=args.width,
            height=args.height,
            num_frames=args.num_frames,
            fps=args.fps,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            spot=not args.no_spot,
            submit=not args.dry_run,
            lora_weights_uri=args.lora_weights_uri,
            lora_scale=args.lora_scale,
        )
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _cmd_submit_ltx_lora_train(args: argparse.Namespace) -> int:
    result = submit_ltx_lora_train(
        LtxLoraTrainSubmitOptions(
            run_id=args.run_id,
            gcs_root=args.gcs_root,
            dataset_dir=Path(args.dataset_dir),
            trigger=args.trigger,
            model_uri=args.model_uri,
            text_encoder_uri=args.text_encoder_uri,
            region=args.region,
            container_image=args.container_image,
            machine_type=args.machine_type,
            accelerator_type=args.accelerator_type,
            accelerator_count=args.accelerator_count,
            boot_disk_type=args.boot_disk_type,
            boot_disk_size_gb=args.boot_disk_size_gb,
            staging_dir=Path(args.staging_dir),
            resolution_bucket=args.resolution_bucket,
            training_steps=args.training_steps,
            trainer_repo=args.trainer_repo,
            trainer_ref=args.trainer_ref,
            validation_prompt=args.validation_prompt,
            hf_token_secret=args.hf_token_secret,
            spot=not args.no_spot,
            submit=not args.dry_run,
        )
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _cmd_scan_publication(args: argparse.Namespace) -> int:
    findings = scan_publication(Path(args.path))
    rows = findings_as_dicts(findings)
    if args.json_out:
        write_json(Path(args.json_out).expanduser().resolve(), rows)
    if findings:
        print(format_findings(findings), file=sys.stderr)
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 1 if any(row["severity"] == "error" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
