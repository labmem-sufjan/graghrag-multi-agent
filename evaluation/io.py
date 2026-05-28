"""Read / write evaluation run artifacts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluation.config import OUTPUTS_DIR
from evaluation.schemas import RunResult


def ensure_outputs_dir() -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUTS_DIR


def save_run_jsonl(results: list[RunResult], pipeline: str, tag: str | None = None) -> Path:
    ensure_outputs_dir()
    ts = tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUTS_DIR / f"run_{pipeline}_{ts}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    return path


def load_run_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def latest_run_path(pipeline: str) -> Path | None:
    ensure_outputs_dir()
    files = sorted(OUTPUTS_DIR.glob(f"run_{pipeline}_*.jsonl"), reverse=True)
    return files[0] if files else None
