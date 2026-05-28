"""Merge retrieval blocks and cap chunks / chars sent to the LLM."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkBlock:
    chunk_id: str
    page: str | int
    text: str
    priority: int = 0


def merge_and_limit_context(
    blocks: list[ChunkBlock],
    *,
    max_chunks: int,
    max_chars_per_chunk: int,
    header: str = "",
) -> tuple[str, list[str]]:
    """Deduplicate by chunk_id (higher priority wins), cap count and per-chunk length."""
    by_id: dict[str, ChunkBlock] = {}
    for b in sorted(blocks, key=lambda x: -x.priority):
        if not b.chunk_id:
            continue
        prev = by_id.get(b.chunk_id)
        if prev is None or b.priority >= prev.priority:
            by_id[b.chunk_id] = b

    ordered = sorted(by_id.keys(), key=lambda cid: -by_id[cid].priority)[:max_chunks]
    lines: list[str] = []
    if header:
        lines.append(header)
    for cid in ordered:
        blk = by_id[cid]
        text = (blk.text or "")[:max_chars_per_chunk]
        lines.append(f"\n[{cid}] (p{blk.page})\n{text}")
    return "\n".join(lines).strip(), ordered
