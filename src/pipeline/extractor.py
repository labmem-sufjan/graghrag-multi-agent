"""Extract entities and relations from chunks via Ollama."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from langchain_core.documents import Document
from config.prompts import (
    EXTRACTION_RETRY_USER,
    EXTRACTION_SYSTEM,
    EXTRACTION_USER_TEMPLATE,
)
from config.settings import settings
from src.tools.llm import extract_message_content, get_chat_llm
from src.tools.neo4j_client import GraphEntity, GraphRelation

logger = logging.getLogger(__name__)

_ENTITY_TYPES = {
    "Company", "Product", "Person", "FinancialMetric",
    "Risk", "Regulation", "Location", "Industry",
}
_RELATION_TYPES = {
    "SUBSIDIARY_OF", "PRODUCES", "HAS_RISK", "REPORTED", "LOCATED_IN",
    "COMPETES_WITH", "REGULATED_BY", "EMPLOYS", "INVESTS_IN",
    "PARTNERS_WITH", "RELATED_TO",
}


@dataclass
class ExtractionResult:
    entities: list[GraphEntity]
    relations: list[GraphRelation]


def _get_llm():
    return get_chat_llm(temperature=0, json_mode=settings.extraction_json_mode)


def _strip_json_noise(raw: str) -> str:
    raw = raw.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if fence:
        raw = fence.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM response does not contain JSON object")
    return raw[start : end + 1]


def _repair_json_text(text: str) -> str:
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    return text


def _parse_entities_relations_fallback(body: str) -> dict | None:
    result: dict = {"entities": [], "relations": []}
    for key in ("entities", "relations"):
        m = re.search(rf'"{key}"\s*:\s*(\[)', body)
        if not m:
            continue
        start = m.start(1)
        depth = 0
        end = -1
        for i in range(start, len(body)):
            if body[i] == "[":
                depth += 1
            elif body[i] == "]":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end > start:
            try:
                result[key] = json.loads(_repair_json_text(body[start:end]))
            except json.JSONDecodeError:
                result[key] = []
    if result["entities"] or result["relations"]:
        return result
    return None


def _parse_json_payload(raw: str) -> dict:
    body = _strip_json_noise(raw)
    last_err: Exception | None = None
    for candidate in (body, _repair_json_text(body)):
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError as e:
            last_err = e
    fallback = _parse_entities_relations_fallback(body)
    if fallback is not None:
        logger.warning("Used fallback JSON parser for entities/relations arrays")
        return fallback
    raise ValueError(f"Invalid JSON from LLM: {last_err}") from last_err


def _build_result(data: dict, chunk_id: str) -> ExtractionResult:
    entities: list[GraphEntity] = []
    seen_names: set[str] = set()
    for item in data.get("entities") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name or name in seen_names:
            continue
        etype = str(item.get("type", "Company")).strip()
        if etype not in _ENTITY_TYPES:
            etype = "Company"
        seen_names.add(name)
        entities.append(GraphEntity(name=name, type=etype))

    relations: list[GraphRelation] = []
    for item in data.get("relations") or []:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "")).strip()
        target = str(item.get("target", "")).strip()
        rel = str(item.get("relation", "RELATED_TO")).strip().upper()
        if not source or not target:
            continue
        if source not in seen_names or target not in seen_names:
            continue
        if rel not in _RELATION_TYPES:
            rel = "RELATED_TO"
        relations.append(
            GraphRelation(
                source=source,
                target=target,
                relation=rel,
                chunk_id=chunk_id,
            )
        )
    return ExtractionResult(entities=entities, relations=relations)


def extract_from_chunk(chunk: Document) -> ExtractionResult:
    chunk_id = chunk.metadata["chunk_id"]
    document_name = chunk.metadata.get("source", "unknown.pdf")
    llm = _get_llm()
    messages: list[tuple[str, str]] = [
        ("system", EXTRACTION_SYSTEM),
        (
            "human",
            EXTRACTION_USER_TEMPLATE.format(
                document_name=document_name,
                chunk_id=chunk_id,
                chunk_text=chunk.page_content,
            ),
        ),
    ]

    last_error: Exception | None = None
    for attempt in range(1, settings.extraction_max_retries + 1):
        try:
            response = llm.invoke(messages)
            content = extract_message_content(response)
            return _build_result(_parse_json_payload(content), chunk_id)
        except (ValueError, json.JSONDecodeError) as e:
            last_error = e
            logger.warning(
                "JSON parse failed for %s (attempt %s/%s): %s",
                chunk_id,
                attempt,
                settings.extraction_max_retries,
                e,
            )
            if attempt < settings.extraction_max_retries:
                messages = [
                    ("system", EXTRACTION_SYSTEM),
                    (
                        "human",
                        EXTRACTION_RETRY_USER.format(
                            document_name=document_name,
                            chunk_id=chunk_id,
                            chunk_text=chunk.page_content,
                        ),
                    ),
                ]

    raise ValueError(
        f"Extraction failed for {chunk_id} after {settings.extraction_max_retries} attempts"
    ) from last_error


def extract_and_load(
    chunks: list[Document],
    graph_client,
    *,
    limit: int | None = None,
    resume: bool = False,
    skip_on_error: bool = True,
) -> dict[str, int]:
    """Extract from chunks and write graph nodes/edges to Neo4j."""
    from src.tools.neo4j_client import Neo4jGraphClient

    if not isinstance(graph_client, Neo4jGraphClient):
        raise TypeError("graph_client must be Neo4jGraphClient")

    total_entities = 0
    total_relations = 0
    failed = 0
    skipped_resume = 0
    subset = chunks[:limit] if limit else chunks

    for i, chunk in enumerate(subset, 1):
        chunk_id = chunk.metadata["chunk_id"]
        if resume and graph_client.chunk_extraction_done(chunk_id):
            skipped_resume += 1
            if i % 20 == 0 or i == len(subset):
                logger.info("Progress %s/%s (skipped resume: %s)", i, len(subset), skipped_resume)
            continue

        if i % 10 == 0 or i == 1 or i == len(subset):
            logger.info("Graph extract %s/%s — %s", i, len(subset), chunk_id)

        extraction_ok = True
        try:
            result = extract_from_chunk(chunk)
        except ValueError as e:
            failed += 1
            extraction_ok = False
            logger.error("%s", e)
            if not skip_on_error:
                raise
            result = ExtractionResult(entities=[], relations=[])

        graph_client.load_extraction(
            chunk_id=chunk_id,
            text=chunk.page_content,
            document_name=chunk.metadata.get("source", "unknown.pdf"),
            page=int(chunk.metadata.get("page", 0)),
            section=chunk.metadata.get("section", ""),
            entities=result.entities,
            relations=result.relations,
            extraction_ok=extraction_ok,
        )
        total_entities += len(result.entities)
        total_relations += len(result.relations)

    return {
        "chunks_processed": len(subset),
        "chunks_skipped_resume": skipped_resume,
        "chunks_failed": failed,
        "entities_extracted": total_entities,
        "relations_extracted": total_relations,
    }
