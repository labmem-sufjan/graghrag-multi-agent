"""Load PDFs, clean text, and split into chunked LangChain Documents."""

from __future__ import annotations

import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import RAW_DOCS_DIR, settings

_SECTION_PATTERN = re.compile(
    r"^(第[一二三四五六七八九十百零\d]+[章节节篇部]|[一二三四五六七八九十]+[、．.])"
)
_WS_PATTERN = re.compile(r"[ \t\u3000]+")
_MULTI_NL = re.compile(r"\n{3,}")


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_PATTERN.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def _detect_section(line: str, current: str | None) -> str | None:
    line = line.strip()
    if len(line) > 80:
        return current
    if _SECTION_PATTERN.match(line):
        return line[:120]
    return current


def load_pdf(path: Path) -> list[Document]:
    """Load a PDF; one Document per page with page metadata."""
    loader = PyPDFLoader(str(path))
    pages = loader.load()
    doc_name = path.name
    for page in pages:
        page.metadata["source"] = doc_name
        page.metadata["source_path"] = str(path)
        page.page_content = _clean_text(page.page_content)
    return pages


def chunk_documents(
    pages: list[Document],
    *,
    doc_stem: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split pages into chunks with stable chunk_id and section hints."""
    if not pages:
        return []

    source_name = pages[0].metadata.get("source", "unknown.pdf")
    stem = doc_stem or Path(source_name).stem

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        separators=["\n\n", "\n", "。", "；", " ", ""],
    )

    chunks: list[Document] = []
    section: str | None = None
    global_idx = 0

    for page in pages:
        page_num = page.metadata.get("page", 0)
        # PyPDFLoader uses 0-based page index in metadata
        display_page = int(page_num) + 1 if isinstance(page_num, int) else page_num

        for line in page.page_content.split("\n"):
            section = _detect_section(line, section)

        for piece in splitter.split_text(page.page_content):
            piece = piece.strip()
            if not piece:
                continue
            chunk_id = f"{stem}_p{display_page:03d}_c{global_idx:04d}"
            global_idx += 1
            chunks.append(
                Document(
                    page_content=piece,
                    metadata={
                        "source": source_name,
                        "source_path": page.metadata.get("source_path", ""),
                        "page": display_page,
                        "chunk_id": chunk_id,
                        "section": section or "",
                        "document_stem": stem,
                    },
                )
            )
    return chunks


def process_pdf(path: Path) -> list[Document]:
    """Load and chunk a single PDF file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    pages = load_pdf(path)
    return chunk_documents(pages, doc_stem=path.stem)


def process_directory(directory: Path | None = None) -> list[Document]:
    """Process all PDF files under raw_docs."""
    directory = Path(directory or RAW_DOCS_DIR)
    pdfs = sorted(directory.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files in {directory}")

    all_chunks: list[Document] = []
    for pdf in pdfs:
        all_chunks.extend(process_pdf(pdf))
    return all_chunks
