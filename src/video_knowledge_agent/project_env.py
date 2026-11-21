"""Read API credentials from the untracked project-root .env file.

No third-party dotenv dependency is required. Explicit API-key arguments take
priority, followed by .env and then the process environment for compatibility.
Only simple KEY=VALUE lines are supported; values are never logged or exported.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_secret(name: str, explicit: str | None = None, *, env_file: Path | None = None) -> str | None:
    if explicit:
        return explicit
    path = env_file if env_file is not None else PROJECT_ROOT / ".env"
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except FileNotFoundError:
        lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition("=")
        if separator and key.strip() == name:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            if value:
                return value
    return os.environ.get(name)
