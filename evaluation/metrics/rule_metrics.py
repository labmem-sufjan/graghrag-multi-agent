"""规则指标：不额外调 LLM，适合快速迭代对比。

must_contain：答案是否含关键词；gold_chunk_recall：检索是否命中标注 chunk。
"""

from __future__ import annotations

import re
from typing import Any

from evaluation.schemas import RunResult

_ENGLISH_PHRASE = re.compile(
    r"\b(Based on|According to|provided text|wholly-owned)\b", re.I
)


def score_run_result(
    result: RunResult,
    must_contain: list[str],
    gold_chunk_ids: list[str] | None = None,
) -> dict[str, Any]:
    answer = result.answer or ""
    retrieved = set(result.chunk_ids or [])
    gold = gold_chunk_ids or []
    scores: dict[str, Any] = {
        "id": result.id,
        "pipeline": result.pipeline,
        "has_answer": bool(answer.strip()),
        "is_chinese": not _ENGLISH_PHRASE.search(answer) and _chinese_ratio(answer) > 0.15,
        "must_contain_hits": 0,
        "must_contain_total": len(must_contain),
        "must_contain_rate": 0.0,
        "gold_chunk_hits": None,
        "gold_chunk_total": len(gold) if gold else None,
        "gold_chunk_recall": None,
        "latency_ms": result.latency_ms,
        "error": result.error,
    }
    if must_contain:
        hits = sum(1 for k in must_contain if k in answer)
        scores["must_contain_hits"] = hits
        scores["must_contain_rate"] = hits / len(must_contain)
    if gold:
        g_hits = sum(1 for cid in gold if cid in retrieved)
        scores["gold_chunk_hits"] = g_hits
        scores["gold_chunk_recall"] = g_hits / len(gold)
    return scores


def _chinese_ratio(text: str) -> float:
    if not text:
        return 0.0
    cn = len(re.findall(r"[\u4e00-\u9fff]", text))
    return cn / max(len(text), 1)


def aggregate_rule_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    n = len(rows)
    gold_rows = [r for r in rows if r.get("gold_chunk_recall") is not None]
    summary: dict[str, float] = {
        "count": n,
        "answer_rate": sum(1 for r in rows if r.get("has_answer")) / n,
        "chinese_rate": sum(1 for r in rows if r.get("is_chinese")) / n,
        "must_contain_rate_avg": sum(r.get("must_contain_rate", 0) for r in rows) / n,
        "latency_ms_avg": sum(r.get("latency_ms", 0) for r in rows) / n,
        "error_rate": sum(1 for r in rows if r.get("error")) / n,
    }
    if gold_rows:
        summary["gold_chunk_recall_avg"] = sum(
            r.get("gold_chunk_recall", 0) for r in gold_rows
        ) / len(gold_rows)
        summary["gold_chunk_items"] = len(gold_rows)
    return summary
