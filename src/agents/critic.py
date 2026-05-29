"""Critic：对生成答案做质检。

默认仅用规则（settings.critic_use_llm=False），依据 document_profile 中的
幻觉模式、控制人/子公司是否出现在 context 但未写入 answer 等。
不通过时目前仅记录 feedback，工作流不会自动重试检索。
"""

from __future__ import annotations

import json
import re

from config.document_profile import get_document_profile
from config.prompts import CRITIC_SYSTEM
from config.settings import settings
from src.agents.state import AgentState
from src.tools.llm import extract_message_content, get_chat_llm

_ENGLISH_TEMPLATE = re.compile(
    r"\b(Based on the provided|According to the)\b", re.I
)


def _rule_based_critic(context: str, answer: str, question: str) -> tuple[bool, str]:
    profile = get_document_profile()
    issues: list[str] = []

    if len(re.findall(r"[A-Za-z]{4,}", answer)) > 15:
        issues.append("回答含大量英文，请使用简体中文")

    for pat in profile.hallucination_patterns:
        if pat.search(answer):
            issues.append("回答含已知幻觉或上下文未支持的人名/表述")
            break

    if _ENGLISH_TEMPLATE.search(answer):
        issues.append("回答以英文模板开头")

    if any(k in question for k in profile.controller_question_keywords):
        in_ctx = profile.controllers_in_context(context)
        if in_ctx and not any(n in answer for n in in_ctx):
            issues.append(
                f"上下文已披露控制人（{'、'.join(in_ctx[:2])}），回答未体现"
            )

    if "子公司" in question or "全资" in question:
        markers = profile.subsidiary_context_markers
        if any(m in context for m in markers):
            in_ctx = profile.subsidiaries_in_context(context)
            if in_ctx and not any(n in answer for n in in_ctx):
                issues.append("上下文含子公司信息，回答未列出相关子公司名称")

    if issues:
        return False, "；".join(issues)
    return True, "规则校验：回答为中文且与上下文关键信息一致"


def _parse_critic_json(raw: str) -> dict | None:
    """解析 LLM Critic 的 JSON；过滤 Ollama 误返回的 HTTP 调试文本。"""
    raw = raw.strip()
    if "status_code" in raw and "result" in raw:
        return None
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(raw[start : end + 1])
        if "passed" in data:
            return data
    except json.JSONDecodeError:
        pass
    return None


def critique_answer(state: AgentState) -> AgentState:
    context = state.get("context", "")
    answer = state.get("answer", "")
    question = state.get("question", "")

    passed, feedback = _rule_based_critic(context, answer, question)

    if not passed and settings.critic_use_llm:
        llm = get_chat_llm(temperature=0, json_mode=True)
        resp = llm.invoke(
            [
                ("system", CRITIC_SYSTEM),
                ("human", f"检索上下文：\n{context[:5000]}\n\n回答：\n{answer}"),
            ]
        )
        raw = extract_message_content(resp)
        data = _parse_critic_json(raw)
        if data is not None:
            passed = bool(data.get("passed", passed))
            fb = str(data.get("feedback", "")).strip()
            if fb and "status_code" not in fb:
                feedback = fb

    return {
        **state,
        "critic_passed": passed,
        "critic_feedback": feedback if not passed else "通过",
    }
