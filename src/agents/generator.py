"""Generate：把检索上下文与用户问题交给 LLM，生成最终答案。

若检测到回答以英文为主，会用 ANSWER_RETRY_USER 再生成一次（见 llm.is_mostly_chinese）。
"""

from __future__ import annotations

from config.prompts import ANSWER_RETRY_USER, ANSWER_SYSTEM
from src.agents.state import AgentState
from src.tools.llm import extract_message_content, get_chat_llm, is_mostly_chinese


def generate_answer(state: AgentState) -> AgentState:
    question = state["question"]
    context = state.get("context", "")
    route = state.get("route", "vector")
    reason = state.get("route_reason", "")

    llm = get_chat_llm(temperature=0)
    prompt = f"""检索策略：{route}（{reason}）

【检索上下文 — 仅据此作答】
{context}

---
【用户问题】
{question}

请用简体中文作答，人名与公司名必须与上文一致。"""

    resp = llm.invoke([("system", ANSWER_SYSTEM), ("human", prompt)])
    answer = extract_message_content(resp)

    if not is_mostly_chinese(answer):
        retry = llm.invoke(
            [
                ("system", ANSWER_SYSTEM),
                (
                    "human",
                    ANSWER_RETRY_USER.format(context=context[:7000], question=question),
                ),
            ]
        )
        answer = extract_message_content(retry)

    return {**state, "answer": answer}
