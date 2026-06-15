from __future__ import annotations

from pathlib import Path

from avatar_video_workbench.publication import scan_publication


def test_scan_rejects_service_account_marker(tmp_path: Path) -> None:
    marker = '{"type": "service_' + 'account"}\n'
    (tmp_path / "secret.json").write_text(marker, encoding="utf-8")

    findings = scan_publication(tmp_path)

    assert any(item.code == "service_account_json" and item.severity == "error" for item in findings)


def test_scan_allows_documented_placeholders(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("output: gs://YOUR_BUCKET/runs/demo\n", encoding="utf-8")

    findings = scan_publication(tmp_path)

    assert findings == []


def test_scan_rejects_generated_media_files(tmp_path: Path) -> None:
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "sample.mp4").write_bytes(b"fake video")

    findings = scan_publication(tmp_path)

    assert any(item.code == "generated_media" and item.severity == "error" for item in findings)
