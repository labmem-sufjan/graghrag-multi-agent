"""Shared Ollama chat client — Chinese-first defaults."""

from __future__ import annotations

import re

from langchain_ollama import ChatOllama

from config.settings import settings

_CHINESE_RATIO_RE = re.compile(r"[\u4e00-\u9fff]")
_ENGLISH_PHRASE_RE = re.compile(
    r"\b(Based on|provided text|According to|the actual controller|"
    r"wholly-owned|Please note|As for the)\b",
    re.I,
)


def get_chat_llm(*, temperature: float = 0, json_mode: bool = False) -> ChatOllama:
    kwargs: dict = {
        "model": settings.ollama_llm_model,
        "base_url": settings.ollama_base_url,
        "temperature": temperature,
        "num_ctx": 8192,
    }
    if json_mode:
        kwargs["format"] = "json"
    return ChatOllama(**kwargs)


def is_mostly_chinese(text: str) -> bool:
    if not text.strip():
        return True
    cn = len(_CHINESE_RATIO_RE.findall(text))
    letters = len(re.findall(r"[A-Za-z]", text))
    if letters > 80 and cn < 30:
        return False
    if _ENGLISH_PHRASE_RE.search(text):
        return False
    return cn >= max(letters // 3, 20) or letters < 40


def extract_message_content(response) -> str:
    content = response.content if hasattr(response, "content") else str(response)
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        content = "\n".join(parts)
    return str(content).strip()
