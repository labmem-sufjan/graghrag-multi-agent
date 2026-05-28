"""CLI entry: chunk PDFs -> Chroma vectors + Neo4j knowledge graph."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow `python -m src.pipeline.build_knowledge` from project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import RAW_DOCS_DIR
from src.pipeline.document_processor import process_directory, process_pdf
from src.pipeline.extractor import extract_and_load
from src.tools.neo4j_client import Neo4jGraphClient
from src.tools.vector_client import index_chunks

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build(
    *,
    source: Path | None,
    limit: int | None,
    skip_chroma: bool,
    skip_graph: bool,
    reset_chroma: bool,
    resume: bool,
) -> None:
    if source and source.is_file():
        chunks = process_pdf(source)
        doc_label = source.name
    else:
        directory = source if source and source.is_dir() else RAW_DOCS_DIR
        chunks = process_directory(directory)
        doc_label = str(directory)

    logger.info("Loaded %s chunks from %s", len(chunks), doc_label)
    if limit:
        chunks = chunks[:limit]
        logger.info("Limited to first %s chunks for this run", limit)

    if not skip_chroma:
        count = index_chunks(chunks, reset=reset_chroma)
        logger.info("Indexed %s chunks in Chroma (collection persist: data/chroma_db)", count)
    else:
        logger.info("Skipped Chroma indexing")

    if not skip_graph:
        with Neo4jGraphClient() as graph:
            graph.ensure_schema()
            seen_docs: set[str] = set()
            for chunk in chunks:
                doc_name = chunk.metadata.get("source", "unknown.pdf")
                if doc_name not in seen_docs:
                    seen_docs.add(doc_name)
                    graph.upsert_document(
                        doc_name,
                        chunk.metadata.get("source_path", ""),
                    )
            stats = extract_and_load(chunks, graph, limit=None, resume=resume)
            logger.info(
                "Graph extraction: %s chunks (%s resumed skip, %s parse failed), "
                "%s entity mentions, %s relations",
                stats["chunks_processed"],
                stats.get("chunks_skipped_resume", 0),
                stats.get("chunks_failed", 0),
                stats["entities_extracted"],
                stats["relations_extracted"],
            )
            if stats.get("chunks_failed"):
                logger.warning(
                    "部分 chunk JSON 抽取失败，已仅写入 Chunk；可 --resume 重试失败项"
                )
            final = graph.stats()
            logger.info(
                "Neo4j totals — documents: %s, chunks: %s, entities: %s, relations: %s",
                final["documents"],
                final["chunks"],
                final["entities"],
                final["relations"],
            )
    else:
        logger.info("Skipped Neo4j graph build")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Chroma chunk index and Neo4j knowledge graph from PDFs.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="PDF file or directory (default: data/raw_docs)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N chunks (useful for dry runs)",
    )
    parser.add_argument("--skip-chroma", action="store_true")
    parser.add_argument("--skip-graph", action="store_true")
    parser.add_argument(
        "--reset-chroma",
        action="store_true",
        help="Delete and recreate the Chroma collection before indexing",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip chunks already extracted in Neo4j (continue after interruption)",
    )
    args = parser.parse_args()
    build(
        source=args.source,
        limit=args.limit,
        skip_chroma=args.skip_chroma,
        skip_graph=args.skip_graph,
        reset_chroma=args.reset_chroma,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
