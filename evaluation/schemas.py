"""Data structures for evaluation runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EvalSample:
    id: str
    question: str
    ground_truth: str
    category: str = ""
    route_hint: str = ""
    must_contain: list[str] = field(default_factory=list)
    gold_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class RunResult:
    id: str
    question: str
    ground_truth: str
    answer: str
    contexts: list[str]
    chunk_ids: list[str]
    pipeline: str
    category: str = ""
    route: str | None = None
    route_reason: str | None = None
    critic_passed: bool | None = None
    critic_feedback: str | None = None
    latency_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
