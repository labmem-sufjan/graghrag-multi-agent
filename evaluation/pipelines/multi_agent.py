"""Full system: LangGraph Router + Experts + Critic."""

from __future__ import annotations

import time

from evaluation.config import PIPELINE_AGENT
from evaluation.schemas import EvalSample, RunResult
from src.graph_workflow import run_query


def _contexts_from_state(context_str: str) -> list[str]:
    if not context_str:
        return []
    parts = context_str.split("\n\n")
    return [p.strip() for p in parts if len(p.strip()) > 80]


def run_multi_agent(sample: EvalSample) -> RunResult:
    t0 = time.perf_counter()
    try:
        state = run_query(sample.question)
        ctx_str = state.get("context", "")
        return RunResult(
            id=sample.id,
            question=sample.question,
            ground_truth=sample.ground_truth,
            answer=state.get("answer", ""),
            contexts=_contexts_from_state(ctx_str) or [ctx_str[:8000]],
            chunk_ids=state.get("chunk_ids", []),
            pipeline=PIPELINE_AGENT,
            category=sample.category,
            route=state.get("route"),
            route_reason=state.get("route_reason"),
            critic_passed=state.get("critic_passed"),
            critic_feedback=state.get("critic_feedback"),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
    except Exception as e:
        return RunResult(
            id=sample.id,
            question=sample.question,
            ground_truth=sample.ground_truth,
            answer="",
            contexts=[],
            chunk_ids=[],
            pipeline=PIPELINE_AGENT,
            category=sample.category,
            latency_ms=(time.perf_counter() - t0) * 1000,
            error=str(e),
        )
