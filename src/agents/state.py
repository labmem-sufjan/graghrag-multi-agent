"""LangGraph 在各节点之间传递的共享状态（TypedDict）。

每个节点读取/写入其中部分字段；最终 invoke 返回的 dict 即完整状态。
"""

from __future__ import annotations

from typing import Literal, TypedDict


class AgentState(TypedDict, total=False):
    question: str  # 用户问题
    route: Literal["vector", "graph", "hybrid"]  # Router 选定的检索策略
    route_reason: str  # 路由依据（规则或 LLM），便于调试与评测记录
    context: str  # 检索专家拼好的上下文，供 Generate 使用
    chunk_ids: list[str]  # 本次检索涉及的 chunk_id，评测算 gold_chunk_recall
    answer: str  # 生成答案
    critic_passed: bool  # Critic 是否通过
    critic_feedback: str  # 未通过时的原因
    retry_count: int  # 预留：未来可做「Critic 不通过则重检索」
