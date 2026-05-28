"""Question-aware keyword expansion for graph / hybrid retrieval."""

from __future__ import annotations

import re

# 问题主题 → 招股书常用检索词
_TOPIC_TERMS: list[tuple[re.Pattern[str], list[str]]] = [
    (
        re.compile(r"实际控制人|控股股东|控制人"),
        ["实际控制人", "控股股东", "王兴兴", "表决权", "23.8216%", "68.7816%"],
    ),
    (
        re.compile(r"全资子公司|子公司"),
        [
            "发行人子公司",
            "全资子公司",
            "系发行人的全资子公司",
            "宇树机器人",
            "上海高羿",
            "北京灵翌",
            "深圳天羿",
            "宁波宇树",
            "重庆宇羿",
            "宇树星盟",
            "UNITREE",
        ],
    ),
    (re.compile(r"营业收入|净利润|毛利率|财务"), ["营业收入", "净利润", "毛利率", "万元"]),
    (re.compile(r"风险"), ["风险因素", "风险"]),
    (re.compile(r"主营业务|做什么|业务"), ["主营业务", "人形机器人", "四足机器人"]),
    (re.compile(r"募资|募集资金"), ["募集资金", "投资项目"]),
    (re.compile(r"股东|股权"), ["股东", "股权", "持股"]),
]


def extract_search_keywords(question: str) -> list[str]:
    """Expand user question to Chinese keywords for Neo4j chunk / entity search."""
    found: list[str] = []
    for pattern, terms in _TOPIC_TERMS:
        if pattern.search(question):
            found.extend(terms)
    for tok in re.split(r"[\s，、？?；;]+", question):
        tok = tok.strip()
        if len(tok) >= 2 and tok not in found:
            found.append(tok)
    if "宇树" not in "".join(found):
        found.append("宇树科技")
    # preserve order, dedupe
    seen: set[str] = set()
    out: list[str] = []
    for k in found:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out[:12]
