"""Hybrid retrieval: vector top-K then graph expansion via chunk entities."""

from __future__ import annotations

from config.settings import settings
from src.agents.state import AgentState
from src.tools.context_limit import ChunkBlock, merge_and_limit_context
from src.tools.neo4j_client import Neo4jGraphClient
from src.tools.retrieval import extract_search_keywords
from src.tools.vector_client import similarity_search


def _chunk_block_from_row(c: dict, *, priority: int) -> ChunkBlock:
    return ChunkBlock(
        chunk_id=c.get("chunk_id", ""),
        page=c.get("page", "?"),
        text=c.get("text") or "",
        priority=priority,
    )


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
                priority=30 - i,
            )
        )

    graph_lines: list[str] = []
    with Neo4jGraphClient() as graph:
        if "子公司" in question or "全资" in question:
            for c in graph.search_subsidiary_chunks(limit=4):
                blocks.append(_chunk_block_from_row(c, priority=90))
        if any(k in question for k in ("控制", "股东", "控股")):
            for c in graph.search_controller_chunks(limit=3):
                blocks.append(_chunk_block_from_row(c, priority=88))
        if "风险" in question:
            for c in graph.search_risk_chunks(limit=5):
                blocks.append(_chunk_block_from_row(c, priority=86))
        for c in graph.search_chunks_by_keywords(
            keywords, boost_phrases=keywords[:6], limit=5
        ):
            blocks.append(_chunk_block_from_row(c, priority=70))

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
