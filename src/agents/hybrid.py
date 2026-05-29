"""Hybrid Expert：Chroma 向量 + Neo4j 定向 chunk + 图谱子图。

向量负责语义召回；图谱负责补足「章节标题对不上 embedding」的段落（如风险因素、子公司表）。
上下文经 context_limit 截断，避免书末无关 chunk 淹没 LLM。
"""

from __future__ import annotations

from config.settings import settings
from src.agents.state import AgentState
from src.tools.context_limit import ChunkBlock, merge_and_limit_context
from src.tools.directed_retrieval import fetch_directed_chunks
from src.tools.neo4j_client import Neo4jGraphClient
from src.tools.retrieval import extract_search_keywords
from src.tools.vector_client import similarity_search


def retrieve_hybrid(state: AgentState) -> AgentState:
    question = state["question"]
    keywords = extract_search_keywords(question)
    blocks: list[ChunkBlock] = []

    docs = similarity_search(question, k=settings.retrieval_top_k)
    for i, doc in enumerate(docs):
        blocks.append(
            ChunkBlock(
                chunk_id=doc.metadata.get("chunk_id", ""),
                page=doc.metadata.get("page", "?"),
                text=doc.page_content,
                priority=30 - i,  # 向量结果优先级低于定向 chunk
            )
        )

    graph_lines: list[str] = []
    graph_chunk_ids: list[str] = []
    with Neo4jGraphClient() as graph:
        for row, priority in fetch_directed_chunks(
            question, graph, limits={"subsidiary": 4, "controller": 3, "risk": 5}
        ):
            blocks.append(
                ChunkBlock(
                    chunk_id=row["chunk_id"],
                    page=row.get("page", "?"),
                    text=row.get("text") or "",
                    priority=priority,
                )
            )

        for c in graph.search_chunks_by_keywords(
            keywords, boost_phrases=keywords[:6], limit=5
        ):
            cid = c["chunk_id"]
            if not any(b.chunk_id == cid for b in blocks):
                blocks.append(
                    ChunkBlock(
                        chunk_id=cid,
                        page=c.get("page", "?"),
                        text=c.get("text") or "",
                        priority=70,
                    )
                )

        chunk_ids_seed = [b.chunk_id for b in blocks if b.chunk_id]
        entity_names = graph.expand_entities_from_chunk_ids(
            chunk_ids_seed, limit=settings.graph_entity_limit
        )
        if not entity_names:
            entity_names = [e["name"] for e in graph.search_entities(question, limit=5)]
        gctx, graph_chunk_ids = graph.get_graph_context(
            entity_names,
            max_chunks=min(settings.graph_hop_chunks, 5),
        )
        if gctx:
            graph_lines.append(gctx)

    keyword_ctx, kw_ids = merge_and_limit_context(
        [b for b in blocks if b.priority >= 70],
        max_chunks=settings.context_max_chunks,
        max_chars_per_chunk=settings.context_chunk_max_chars,
        header="【关键词/定向片段】",
    )
    vector_ctx, vec_ids = merge_and_limit_context(
        [b for b in blocks if 20 <= b.priority < 70],
        max_chunks=min(settings.retrieval_top_k, settings.context_max_chunks),
        max_chars_per_chunk=settings.context_chunk_max_chars,
        header="【向量检索片段】",
    )

    parts = [p for p in (keyword_ctx, vector_ctx) if p]
    if graph_lines:
        parts.append("\n".join(graph_lines)[: settings.context_chunk_max_chars * 3])
    context = "\n\n".join(parts)
    chunk_ids = list(dict.fromkeys(kw_ids + vec_ids + graph_chunk_ids))[
        : settings.context_max_chunks
    ]

    return {**state, "context": context, "chunk_ids": chunk_ids}
