"""Evaluation pipelines (retrieval + generation strategies)."""

from evaluation.pipelines.multi_agent import run_multi_agent
from evaluation.pipelines.naive_rag import run_naive_rag

__all__ = ["run_naive_rag", "run_multi_agent"]
