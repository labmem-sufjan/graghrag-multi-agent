"""Vector retrieval expert — Chroma similarity search."""

from __future__ import annotations

from config.settings import settings
from src.agents.state import AgentState
from src.tools.vector_client import similarity_search


def retrieve_vector(state: AgentState) -> AgentState:
    question = state["question"]
    docs = similarity_search(question, k=settings.retrieval_top_k)
    lines = ["【向量检索片段】"]
    chunk_ids: list[str] = []
    for i, doc in enumerate(docs, 1):
        cid = doc.metadata.get("chunk_id", f"doc_{i}")
        chunk_ids.append(cid)
        page = doc.metadata.get("page", "?")
        lines.append(f"\n[{cid}] (p{page})\n{doc.page_content}")

    context = "\n".join(lines) if docs else "【向量检索】未找到相关片段。"
    return {**state, "context": context, "chunk_ids": chunk_ids}
