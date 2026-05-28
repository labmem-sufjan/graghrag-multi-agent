"""Shared answer generation for evaluation baselines."""

from __future__ import annotations

from config.prompts import ANSWER_SYSTEM
from src.tools.llm import extract_message_content, get_chat_llm


def generate_answer_from_context(
    question: str,
    context: str,
    *,
    pipeline_label: str = "naive_rag",
) -> str:
    llm = get_chat_llm(temperature=0)
    prompt = f"""检索策略：{pipeline_label}

【检索上下文 — 仅据此作答】
{context}

---
【用户问题】
{question}

请用简体中文作答，人名与公司名必须与上文一致。"""
    resp = llm.invoke([("system", ANSWER_SYSTEM), ("human", prompt)])
    return extract_message_content(resp)
