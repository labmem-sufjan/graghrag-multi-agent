"""Load and validate test_dataset.json."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.config import DATASET_PATH
from evaluation.schemas import EvalSample


def load_dataset(path: Path | None = None) -> list[EvalSample]:
    path = path or DATASET_PATH
    raw = json.loads(path.read_text(encoding="utf-8"))
    samples = []
    for item in raw.get("samples", []):
        samples.append(
            EvalSample(
                id=item["id"],
                question=item["question"],
                ground_truth=item["ground_truth"],
                category=item.get("category", ""),
                route_hint=item.get("route_hint", ""),
                must_contain=item.get("must_contain", []),
                gold_chunk_ids=item.get("gold_chunk_ids", []),
            )
        )
    return samples
