"""LangGraph shared state for multi-agent QA."""

from __future__ import annotations

from typing import Literal, TypedDict


class AgentState(TypedDict, total=False):
    question: str
    route: Literal["vector", "graph", "hybrid"]
    route_reason: str
    context: str
    chunk_ids: list[str]
    answer: str
    critic_passed: bool
    critic_feedback: str
    retry_count: int
