"""Graph retrieval expert — keywords + entities + subgraph."""

from __future__ import annotations

from config.settings import settings
from src.agents.state import AgentState
from src.tools.context_limit import ChunkBlock, merge_and_limit_context
from src.tools.neo4j_client import Neo4jGraphClient
from src.tools.retrieval import extract_search_keywords


def retrieve_graph(state: AgentState) -> AgentState:
    question = state["question"]
    keywords = extract_search_keywords(question)
    blocks: list[ChunkBlock] = []
    names: list[str] = []
    extra_lines: list[str] = []

    with Neo4jGraphClient() as graph:
        if "子公司" in question or "全资" in question:
            for c in graph.search_subsidiary_chunks(limit=6):
                blocks.append(
                    ChunkBlock(
                        chunk_id=c["chunk_id"],
                        page=c.get("page", "?"),
                        text=c.get("text") or "",
                        priority=90,
                    )
                )
        if any(k in question for k in ("控制", "股东", "控股")):
            for c in graph.search_controller_chunks(limit=4):
                blocks.append(
                    ChunkBlock(
                        chunk_id=c["chunk_id"],
                        page=c.get("page", "?"),
                        text=c.get("text") or "",
                        priority=88,
                    )
                )
        if "风险" in question:
            for c in graph.search_risk_chunks(limit=6):
                blocks.append(
                    ChunkBlock(
                        chunk_id=c["chunk_id"],
                        page=c.get("page", "?"),
                        text=c.get("text") or "",
                        priority=86,
                    )
                )
        for c in graph.search_chunks_by_keywords(
            keywords,
            boost_phrases=keywords[:6],
            limit=max(settings.graph_hop_chunks, 8),
        ):
            blocks.append(
                ChunkBlock(
                    chunk_id=c["chunk_id"],
                    page=c.get("page", "?"),
                    text=c.get("text") or "",
                    priority=70,
                )
            )

        for kw in keywords[:6]:
            for e in graph.search_entities(kw, limit=4):
                if e["name"] not in names:
                    names.append(e["name"])

        if "王兴兴" not in names and any(
            k in question for k in ("控制", "股东", "控股")
        ):
            names.append("王兴兴")
        if "宇树科技股份有限公司" not in names:
            names.insert(0, "宇树科技股份有限公司")

        graph_ctx, graph_cids = graph.get_graph_context(
            names[:12],
            max_relations=20,
            max_chunks=min(settings.graph_hop_chunks, 5),
        )
        if graph_ctx.strip():
            extra_lines.append(graph_ctx[: settings.context_chunk_max_chars * 3])

        if "子公司" in question or "全资" in question:
            subs = graph.search_subsidiary_relations()
            if subs:
                rel_lines = ["\n【图谱：子公司关系】"]
                for s in subs[:12]:
                    rel_lines.append(
                        f"- ({s['src']})-[:SUBSIDIARY_OF]->({s['tgt']})"
                    )
                extra_lines.append("\n".join(rel_lines))

    kw_ctx, chunk_ids = merge_and_limit_context(
        blocks,
        max_chunks=settings.context_max_chunks,
        max_chars_per_chunk=settings.context_chunk_max_chars,
        header="【关键词匹配的文档片段】",
    )
    all_chunk_ids = list(
        dict.fromkeys(chunk_ids + graph_cids)
    )[: settings.context_max_chunks]
    header = "【检索关键词】" + "、".join(keywords[:8])
    parts = [header, kw_ctx] if kw_ctx else [header, "【图谱检索】未找到相关文档片段。"]
    if extra_lines:
        parts.extend(extra_lines)
    context = "\n\n".join(p for p in parts if p)

    return {**state, "context": context, "chunk_ids": all_chunk_ids}
