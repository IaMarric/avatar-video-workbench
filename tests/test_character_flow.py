from __future__ import annotations

import json
from pathlib import Path

import yaml
from PIL import Image

from avatar_video_workbench.character_flow import (
    CharacterDatasetOptions,
    build_ltx_lora_training_config,
    create_character_dataset,
)


def test_create_character_dataset_writes_ltx_training_inputs(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (96, 128), "white").save(source)

    def fake_generator(_source_image: Path, _prompt: str, index: int) -> Image.Image:
        return Image.new("RGB", (128, 128), (20 * index, 100, 140))

    manifest = create_character_dataset(
        CharacterDatasetOptions(
            source_image=source,
            out_dir=tmp_path / "run",
            trigger="avwpirate person",
            variant_count=3,
            ltx_model_path="/models/ltx.safetensors",
            ltx_text_encoder_path="/models/gemma",
        ),
        image_generator=fake_generator,
    )

    run_dir = tmp_path / "run"
    assert manifest["variant_count"] == 3
    assert (run_dir / "dataset" / "images" / "001.png").is_file()
    assert "avwpirate person" in (run_dir / "dataset" / "images" / "001.txt").read_text(encoding="utf-8")
    trainer_rows = json.loads((run_dir / "dataset.json").read_text(encoding="utf-8"))
    assert len(trainer_rows) == 3
    assert trainer_rows[0]["media_path"] == "dataset/images/001.png"
    legacy_rows = json.loads((run_dir / "ltx_trainer" / "dataset.json").read_text(encoding="utf-8"))
    assert legacy_rows[0]["media_path"] == "../dataset/images/001.png"
    config = yaml.safe_load((run_dir / "configs" / "ltx_lora_training.yaml").read_text(encoding="utf-8"))
    assert config["model"]["training_mode"] == "lora"
    assert config["training_strategy"]["first_frame_conditioning_p"] == 0.75
    assert config["data"]["preprocessed_data_root"] == ".precomputed"


def test_build_ltx_lora_training_config_uses_concrete_paths() -> None:
    config = build_ltx_lora_training_config(
        model_path="/models/ltx.safetensors",
        text_encoder_path="/models/gemma",
        preprocessed_data_root="/data/.precomputed",
        output_dir="/out",
        validation_prompt="avwpirate person stands on a ship deck",
        steps=800,
    )

    assert config["model"]["model_path"] == "/models/ltx.safetensors"
    assert config["model"]["text_encoder_path"] == "/models/gemma"
    assert config["optimization"]["steps"] == 800
    assert config["validation"]["prompts"] == ["avwpirate person stands on a ship deck"]
