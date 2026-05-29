"""按问题类型从 Neo4j 拉「定向 chunk」，优先于纯向量排序。

例如：问子公司 → search_subsidiary_chunks；问风险 → search_risk_chunks。
graph_expert 与 hybrid_expert 共用，避免两处复制 if-else。
"""

from __future__ import annotations

from config.document_profile import DocumentProfile, get_document_profile
from src.tools.neo4j_client import Neo4jGraphClient

# (内部标志名, Neo4jGraphClient 方法名, 优先级数字越大越先保留)
_DIRECTED_SPECS: list[tuple[str, str, int]] = [
    ("subsidiary", "search_subsidiary_chunks", 90),
    ("controller", "search_controller_chunks", 88),
    ("risk", "search_risk_chunks", 86),
    ("financial", "search_financial_chunks", 85),
    ("issuer_profile", "search_issuer_profile_chunks", 84),
]


def _question_flags(question: str) -> dict[str, bool]:
    """根据问句关键词判断需要启用哪些定向检索。"""
    return {
        "subsidiary": "子公司" in question or "全资" in question,
        "controller": any(
            k in question for k in ("控制", "股东", "控股", "法定代表人")
        ),
        "risk": "风险" in question,
        "financial": any(
            k in question
            for k in (
                "营收",
                "收入",
                "净利润",
                "毛利",
                "扣非",
                "财务",
                "万元",
            )
        ),
        "issuer_profile": any(
            k in question for k in ("成立", "注册地址", "基本情况", "注册资本")
        ),
    }


def fetch_directed_chunks(
    question: str,
    graph: Neo4jGraphClient,
    *,
    profile: DocumentProfile | None = None,
    limits: dict[str, int] | None = None,
) -> list[tuple[dict, int]]:
    """返回 [(chunk 行 dict, priority), ...]，供 ChunkBlock 组装。"""
    profile = profile or get_document_profile()
    flags = _question_flags(question)
    limits = limits or {}
    out: list[tuple[dict, int]] = []
    seen: set[str] = set()

    for key, method_name, priority in _DIRECTED_SPECS:
        if not flags.get(key):
            continue
        method = getattr(graph, method_name)
        limit = limits.get(key, 6 if key != "controller" else 4)
        for row in method(limit=limit, profile=profile):
            cid = row.get("chunk_id", "")
            if cid and cid not in seen:
                seen.add(cid)
                out.append((row, priority))
    return out
