"""Baseline B0: Chroma Top-K + LLM (no graph, no multi-agent)."""

from __future__ import annotations

import time

from config.settings import settings
from evaluation.config import PIPELINE_NAIVE
from evaluation.pipelines._generate import generate_answer_from_context
from evaluation.schemas import EvalSample, RunResult
from src.tools.vector_client import similarity_search


def _format_context(docs) -> tuple[str, list[str], list[str]]:
    lines = ["【向量检索片段】"]
    chunk_ids: list[str] = []
    contexts: list[str] = []
    for doc in docs:
        cid = doc.metadata.get("chunk_id", "")
        if cid:
            chunk_ids.append(cid)
        text = doc.page_content
        contexts.append(text)
        page = doc.metadata.get("page", "?")
        lines.append(f"\n[{cid}] (p{page})\n{text}")
    return "\n".join(lines), chunk_ids, contexts


def run_naive_rag(sample: EvalSample) -> RunResult:
    t0 = time.perf_counter()
    try:
        docs = similarity_search(sample.question, k=settings.retrieval_top_k)
        context, chunk_ids, contexts = _format_context(docs)
        if not docs:
            context = "【向量检索】未找到相关片段。"
        answer = generate_answer_from_context(
            sample.question,
            context,
            pipeline_label=PIPELINE_NAIVE,
        )
        return RunResult(
            id=sample.id,
            question=sample.question,
            ground_truth=sample.ground_truth,
            answer=answer,
            contexts=contexts,
            chunk_ids=chunk_ids,
            pipeline=PIPELINE_NAIVE,
            category=sample.category,
            route="vector",
            route_reason="baseline: chroma only",
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
            pipeline=PIPELINE_NAIVE,
            category=sample.category,
            latency_ms=(time.perf_counter() - t0) * 1000,
            error=str(e),
        )
