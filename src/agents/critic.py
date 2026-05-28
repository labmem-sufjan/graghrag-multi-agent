"""Critic agent — rule checks first, optional LLM validation."""

from __future__ import annotations

import json
import re

from config.prompts import CRITIC_SYSTEM
from config.settings import settings
from src.agents.state import AgentState
from src.tools.llm import extract_message_content, get_chat_llm

# 常见 Llama 幻觉人名（招股书实际控制人为王兴兴）
_WRONG_CONTROLLER_NAMES = re.compile(r"张阳光|Zhang Yangguang|Yangguang Zhang", re.I)


def _rule_based_critic(context: str, answer: str, question: str) -> tuple[bool, str]:
    issues: list[str] = []

    if len(re.findall(r"[A-Za-z]{4,}", answer)) > 15:
        issues.append("回答含大量英文，请使用简体中文")

    if _WRONG_CONTROLLER_NAMES.search(answer):
        issues.append("出现上下文未支持的「张阳光」等错误人名")

    if any(k in question for k in ("控制", "实际控制人", "控股股东")):
        if "王兴兴" in context and "王兴兴" not in answer:
            issues.append("上下文有王兴兴，回答未体现实际控制人")

    if "子公司" in question or "全资" in question:
        markers = ["全资子公司", "系发行人的全资子公司", "宇树机器人"]
        if any(m in context for m in markers):
            if not any(
                s in answer
                for s in (
                    "宇树机器人",
                    "上海高羿",
                    "北京灵翌",
                    "深圳天羿",
                    "宁波宇树",
                    "重庆宇羿",
                    "宇树星盟",
                    "UNITREE",
                )
            ):
                issues.append("未列出上下文中的全资子公司名称")

    if "Based on the provided" in answer or "According to the" in answer:
        issues.append("回答以英文模板开头")

    if issues:
        return False, "；".join(issues)
    return True, "规则校验：回答为中文且与上下文关键信息一致"


def _parse_critic_json(raw: str) -> dict | None:
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
