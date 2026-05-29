"""评测用数据结构：单题样本 + 单次运行结果。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EvalSample:
    id: str
    question: str
    ground_truth: str  # 参考答案，供 Ragas answer_correctness
    category: str = ""  # factual / governance / financial 等，便于分组统计
    route_hint: str = ""  # 期望路由，仅文档用，不参与自动打分
    must_contain: list[str] = field(default_factory=list)  # 答案应包含的关键词
    gold_chunk_ids: list[str] = field(default_factory=list)  # 期望检索命中的 chunk


@dataclass
class RunResult:
    id: str
    question: str
    ground_truth: str
    answer: str
    contexts: list[str]  # 切分后的上下文片段，供 Ragas
    chunk_ids: list[str]  # 检索返回的 id，供 gold_chunk_recall
    pipeline: str  # naive_rag | multi_agent
    category: str = ""
    route: str | None = None
    route_reason: str | None = None
    critic_passed: bool | None = None
    critic_feedback: str | None = None
    latency_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
