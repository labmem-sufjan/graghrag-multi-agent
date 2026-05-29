"""从 YAML 加载当前发行人配置（document_profile.yml）。

换招股书时改 YAML 即可更新：控制人、子公司列表、幻觉拦截、检索扩展词。
代码通过 get_document_profile() 读取，避免在 agents 里写死公司名。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from config.settings import PROJECT_ROOT, settings

_DEFAULT_PATH = PROJECT_ROOT / "config" / "document_profile.yml"


@dataclass
class TopicExpansion:
    pattern: re.Pattern[str]
    terms: list[str]


@dataclass
class DocumentProfile:
    document_id: str
    issuer_legal_name: str
    issuer_short_names: list[str]
    graph_parent_hint: str
    controller_names: list[str]
    controller_question_keywords: list[str]
    subsidiary_names: list[str]
    subsidiary_context_markers: list[str]
    hallucination_patterns: list[re.Pattern[str]]
    topic_expansions: list[TopicExpansion]

    def default_issuer_keyword(self) -> str:
        return self.issuer_short_names[0] if self.issuer_short_names else self.issuer_legal_name

    def subsidiary_search_keywords(self) -> list[str]:
        generic = [
            "发行人子公司",
            "全资子公司",
            "系发行人的全资子公司",
        ]
        return _dedupe(generic + self.subsidiary_names[:6])

    def controller_search_keywords(self) -> list[str]:
        generic = ["实际控制人", "控股股东", "表决权"]
        return _dedupe(generic + self.controller_names)

    def financial_search_keywords(self) -> list[str]:
        return [
            "营业收入",
            "净利润",
            "扣除非经常性损益",
            "毛利率",
            "万元",
            "合并利润表",
        ]

    def issuer_profile_search_keywords(self) -> list[str]:
        return _dedupe(
            [
                "发行人基本情况",
                "成立日期",
                "注册地址",
                "注册资本",
            ]
            + self.issuer_short_names
        )

    def seed_entity_names(self, question: str) -> list[str]:
        names: list[str] = [self.issuer_legal_name]
        if any(k in question for k in self.controller_question_keywords):
            names.extend(self.controller_names)
        return _dedupe(names)

    def subsidiaries_in_context(self, context: str) -> list[str]:
        return [n for n in self.subsidiary_names if n in context]

    def controllers_in_context(self, context: str) -> list[str]:
        return [n for n in self.controller_names if n in context]


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _compile_patterns(raw: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(p, re.I) for p in raw if p]


def load_document_profile(path: Path | None = None) -> DocumentProfile:
    path = path or Path(settings.document_profile_path)
    if not path.is_file():
        raise FileNotFoundError(f"Document profile not found: {path}")

    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    issuer = data.get("issuer") or {}
    controller = data.get("controller") or {}
    subs = data.get("subsidiaries") or {}

    expansions: list[TopicExpansion] = []
    for item in data.get("topic_expansions") or []:
        pat = item.get("pattern", "")
        if not pat:
            continue
        expansions.append(
            TopicExpansion(
                pattern=re.compile(pat),
                terms=list(item.get("terms") or []),
            )
        )

    return DocumentProfile(
        document_id=str(data.get("document_id", "default")),
        issuer_legal_name=str(issuer.get("legal_name", "")),
        issuer_short_names=list(issuer.get("short_names") or []),
        graph_parent_hint=str(issuer.get("graph_parent_hint", "")),
        controller_names=list(controller.get("names") or []),
        controller_question_keywords=list(controller.get("question_keywords") or []),
        subsidiary_names=list(subs.get("names") or []),
        subsidiary_context_markers=list(subs.get("context_markers") or []),
        hallucination_patterns=_compile_patterns(
            list(data.get("hallucination_patterns") or [])
        ),
        topic_expansions=expansions,
    )


@lru_cache(maxsize=1)
def get_document_profile() -> DocumentProfile:
    return load_document_profile()
