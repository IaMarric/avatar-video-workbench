from __future__ import annotations

from pathlib import Path

from avatar_video_workbench.publication import scan_publication


def test_scan_rejects_service_account_marker(tmp_path: Path) -> None:
    marker = '{"type": "service_' + 'account"}\n'
    (tmp_path / "secret.json").write_text(marker, encoding="utf-8")

    findings = scan_publication(tmp_path)

    assert any(item.code == "service_account_json" and item.severity == "error" for item in findings)


def test_scan_rejects_cloud_storage_uris(tmp_path: Path) -> None:
    uri = "gs://" + "private-bucket/runs/demo\n"
    (tmp_path / "config.yaml").write_text(f"output: {uri}", encoding="utf-8")

    findings = scan_publication(tmp_path)

    assert any(item.code == "gcs_uri" and item.severity == "error" for item in findings)


def test_scan_rejects_generated_media_files(tmp_path: Path) -> None:
    (tmp_path / "sample.mp4").write_bytes(b"fake video")

    findings = scan_publication(tmp_path)

    assert any(item.code == "generated_media" and item.severity == "error" for item in findings)
