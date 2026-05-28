"""Route questions to vector, graph, or hybrid retrieval."""

from __future__ import annotations

import re

from config.prompts import ROUTER_SYSTEM
from src.tools.llm import extract_message_content, get_chat_llm
from src.agents.state import AgentState

_GRAPH_KEYWORDS = re.compile(
    r"子公司|参股|控股|股东|实际控制人|关系|架构|持股|投资|"
    r"关联方|产业链|上游|下游|图谱"
)
_META_KEYWORDS = re.compile(
    r"这个系统|你是干什么的|你能做什么|你是谁|如何使用|什么功能"
)
_RISK_KEYWORDS = re.compile(
    r"风险因素|招股书.*风险|列举.*风险|相关风险|有哪些.*风险|风险[？?]"
)


def _route_by_keywords(question: str) -> str | None:
    if _META_KEYWORDS.search(question):
        return "vector"
    if _RISK_KEYWORDS.search(question) or (
        "风险" in question and any(k in question for k in ("列举", "哪些", "至少"))
    ):
        return "hybrid"
    if _GRAPH_KEYWORDS.search(question):
        # 同时问控制人 + 子公司 → hybrid（向量+图谱更稳）
        if ("控制" in question or "股东" in question) and (
            "子公司" in question or "全资" in question
        ):
            return "hybrid"
        if len(question) > 40:
            return "hybrid"
        return "graph"
    return None


def route_question(state: AgentState) -> AgentState:
    question = state["question"]
    heuristic = _route_by_keywords(question)
    if heuristic:
        return {
            **state,
            "route": heuristic,
            "route_reason": f"关键词规则 → {heuristic}",
        }

    llm = get_chat_llm(temperature=0)
    resp = llm.invoke([("system", ROUTER_SYSTEM), ("human", question)])
    text = extract_message_content(resp).lower()
    if "hybrid" in text:
        route = "hybrid"
    elif "graph" in text:
        route = "graph"
    else:
        route = "vector"
    return {
        **state,
        "route": route,
        "route_reason": f"LLM 路由 → {route}",
    }
