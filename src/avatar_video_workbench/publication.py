from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_EXCLUDES = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "dist",
    "build",
}

DISALLOWED_PUBLIC_EXTENSIONS = {
    ".ckpt": "model_checkpoint",
    ".gif": "generated_media",
    ".jpeg": "image_asset",
    ".jpg": "image_asset",
    ".mov": "generated_media",
    ".mp4": "generated_media",
    ".pth": "model_checkpoint",
    ".pt": "model_checkpoint",
    ".png": "image_asset",
    ".safetensors": "model_checkpoint",
    ".webm": "generated_media",
    ".webp": "image_asset",
}

SECRET_PATTERNS = [
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |)PRIVATE KEY-----")),
    ("service_account_json", re.compile(r'"type"\s*:\s*"service_account"')),
    ("api_token_assignment", re.compile(r"(?i)(api[_-]?key|api[_-]?token|secret|password)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}")),
    ("gcs_uri", re.compile(r"\bgs://[a-z0-9._-]+/[^\s'\"\)]+")),
    ("absolute_home_path", re.compile(r"/home/[A-Za-z0-9._-]+/")),
]

GCP_PROJECT_CONTEXT = re.compile(
    r"(?i)(project_id|project-id|GOOGLE_CLOUD_PROJECT|gcloud config set project)\s*[:= ]+\s*['\"]?([a-z][a-z0-9-]{5,30})"
)

TEXT_EXTENSIONS = {
    ".cfg",
    ".css",
    ".csv",
    ".env",
    ".html",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


@dataclass(frozen=True)
class Finding:
    severity: str
    code: str
    path: str
    line: int
    excerpt: str


def scan_publication(root: Path) -> list[Finding]:
    root = root.expanduser().resolve()
    findings: list[Finding] = []
    for path in _iter_candidate_files(root):
        rel = path.relative_to(root).as_posix()
        artifact_code = DISALLOWED_PUBLIC_EXTENSIONS.get(path.suffix.lower())
        if artifact_code:
            findings.append(
                Finding(
                    "error",
                    artifact_code,
                    rel,
                    0,
                    "Generated media, datasets, and model artifacts stay outside git.",
                )
            )

    for path in _iter_text_files(root):
        rel = path.relative_to(root).as_posix()
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            findings.append(Finding("error", "unreadable_file", rel, 0, str(exc)))
            continue
        for idx, line in enumerate(lines, start=1):
            for code, pattern in SECRET_PATTERNS:
                if _is_safe_placeholder(line):
                    continue
                if pattern.search(line):
                    findings.append(Finding("error", code, rel, idx, _excerpt(line)))
            if not _is_safe_placeholder(line) and GCP_PROJECT_CONTEXT.search(line):
                findings.append(Finding("error", "gcp_project_context", rel, idx, _excerpt(line)))
    return findings


def findings_as_dicts(findings: Iterable[Finding]) -> list[dict[str, object]]:
    return [
        {
            "severity": item.severity,
            "code": item.code,
            "path": item.path,
            "line": item.line,
            "excerpt": item.excerpt,
        }
        for item in findings
    ]


def _iter_candidate_files(root: Path) -> Iterable[Path]:
    root = root.expanduser().resolve()
    for path in sorted(root.rglob("*")):
        if any(part in DEFAULT_EXCLUDES or part.endswith(".egg-info") for part in path.relative_to(root).parts[:-1]):
            continue
        if not path.is_file():
            continue
        yield path


def _iter_text_files(root: Path) -> Iterable[Path]:
    for path in _iter_candidate_files(root):
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        yield path


def _excerpt(line: str) -> str:
    line = line.strip()
    if len(line) <= 140:
        return line
    return line[:137] + "..."


def _is_safe_placeholder(line: str) -> bool:
    placeholders = [
        "YOUR_BUCKET",
        "YOUR_PROJECT",
        "YOUR_PROJECT_ID",
        "PROJECT/REPOSITORY",
        "REGION-docker.pkg.dev/PROJECT",
    ]
    return any(marker in line for marker in placeholders)
