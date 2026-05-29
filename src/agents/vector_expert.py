"""Vector Expert：仅使用 Chroma 语义相似度检索。

适合主营业务、财务数字等「表述接近即可命中」的问题；
治理/子公司/风险等更依赖章节标题的题由 graph/hybrid 处理。
"""

from __future__ import annotations

from config.settings import settings
from src.agents.state import AgentState
from src.tools.vector_client import similarity_search


def retrieve_vector(state: AgentState) -> AgentState:
    question = state["question"]
    docs = similarity_search(question, k=settings.retrieval_top_k)

    lines = ["【向量检索片段】"]
    chunk_ids: list[str] = []
    for doc in docs:
        cid = doc.metadata.get("chunk_id", "")
        if cid:
            chunk_ids.append(cid)
        page = doc.metadata.get("page", "?")
        lines.append(f"\n[{cid}] (p{page})\n{doc.page_content}")

    context = "\n".join(lines) if docs else "【向量检索】未找到相关片段。"
    return {**state, "context": context, "chunk_ids": chunk_ids}
