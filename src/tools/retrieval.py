"""把用户问题扩展成 Neo4j 全文检索用的中文关键词。

领域通用词来自 document_profile.topic_expansions；
公司实体名（控制人、子公司）在匹配到对应问题类型时从 profile 注入。
"""

from __future__ import annotations

import re

from config.document_profile import get_document_profile


def extract_search_keywords(question: str) -> list[str]:
    profile = get_document_profile()
    found: list[str] = []

    for expansion in profile.topic_expansions:
        if expansion.pattern.search(question):
            found.extend(expansion.terms)

    if profile.topic_expansions:
        if re.search(r"实际控制人|控股股东|控制人", question):
            found.extend(profile.controller_names)
        if re.search(r"全资子公司|子公司", question):
            found.extend(profile.subsidiary_names[:8])

    for tok in re.split(r"[\s，、？?；;]+", question):
        tok = tok.strip()
        if len(tok) >= 2 and tok not in found:
            found.append(tok)

    default_kw = profile.default_issuer_keyword()
    if default_kw and default_kw not in "".join(found):
        found.append(default_kw)

    seen: set[str] = set()
    out: list[str] = []
    for k in found:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out[:12]
