"""Ragas metrics wrapper (optional — requires ragas + datasets)."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def run_ragas_evaluation(records: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Run Ragas on completed run records.
    Each record needs: question, answer, contexts (list[str]), ground_truth.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import answer_correctness, answer_relevancy, faithfulness
    except ImportError as e:
        logger.warning("Ragas not available: %s", e)
        return {"error": "ragas or datasets not installed", "skipped": True}

    from src.tools.llm import get_chat_llm
    from src.tools.vector_client import get_embeddings

    rows = [
        r
        for r in records
        if r.get("answer") and not r.get("error")
    ]
    if not rows:
        return {"error": "no valid records for ragas", "skipped": True}

    dataset = Dataset.from_dict(
        {
            "question": [r["question"] for r in rows],
            "answer": [r["answer"] for r in rows],
            "contexts": [r.get("contexts") or [""] for r in rows],
            "ground_truth": [r.get("ground_truth", "") for r in rows],
        }
    )

    llm = LangchainLLMWrapper(get_chat_llm(temperature=0))
    embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, answer_correctness],
        llm=llm,
        embeddings=embeddings,
    )

    if hasattr(result, "to_pandas"):
        df = result.to_pandas()
        summary = {col: float(df[col].mean()) for col in df.columns if df[col].dtype.kind in "fc"}
        return {"summary": summary, "per_row": df.to_dict(orient="records")}
    if isinstance(result, dict):
        return {"summary": result}
    return {"raw": str(result)}
