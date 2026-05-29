"""离线流水线第一步：PDF → 清洗 → 分块。

每个 chunk 带有稳定的 chunk_id（评测 gold_chunk、Neo4j、Chroma 都依赖它），
并尽量附带当前章节标题 section，供后续图谱抽取与检索排序使用。
"""

from __future__ import annotations

import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import RAW_DOCS_DIR, settings

# 匹配「第一章」「一、」等章节标题行（用于 section 元数据，不切断正文）
_SECTION_PATTERN = re.compile(
    r"^(第[一二三四五六七八九十百零\d]+[章节节篇部]|[一二三四五六七八九十]+[、．.])"
)
# 合并多余空白，避免 PDF 抽取出的乱码空格
_WS_PATTERN = re.compile(r"[ \t\u3000]+")
_MULTI_NL = re.compile(r"\n{3,}")


def _clean_text(text: str) -> str:
    """统一换行、压缩空白，减少分块噪声。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_PATTERN.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def _detect_section(line: str, current: str | None) -> str | None:
    """逐行扫描：遇到像章节标题的短行则更新 section，否则保持上一节。"""
    line = line.strip()
    if len(line) > 80:
        return current
    if _SECTION_PATTERN.match(line):
        return line[:120]
    return current


def load_pdf(path: Path) -> list[Document]:
    """按页加载 PDF，每页一个 LangChain Document。

    metadata 含 source、source_path、page（0-based，分块时会 +1 展示用）。
    """
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
    """将多页文档切成带 chunk_id 的片段。

    chunk_id 格式：{文件名stem}_p{页码3位}_c{全局序号4位}
    例：yushu_p057_c0084 —— 与 test_dataset 里 gold_chunk_ids 对齐。
    """
    if not pages:
        return []

    source_name = pages[0].metadata.get("source", "unknown.pdf")
    stem = doc_stem or Path(source_name).stem

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        # 中文招股书优先按段落、句号切，避免半句话
        separators=["\n\n", "\n", "。", "；", " ", ""],
    )

    chunks: list[Document] = []
    section: str | None = None
    global_idx = 0

    for page in pages:
        page_num = page.metadata.get("page", 0)
        # PyPDFLoader 的 page 从 0 开始，对外展示用 1-based
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
    """单文件入口：加载 + 分块。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    pages = load_pdf(path)
    return chunk_documents(pages, doc_stem=path.stem)


def process_directory(directory: Path | None = None) -> list[Document]:
    """处理 data/raw_docs 下全部 PDF，合并所有 chunk。"""
    directory = Path(directory or RAW_DOCS_DIR)
    pdfs = sorted(directory.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files in {directory}")

    all_chunks: list[Document] = []
    for pdf in pdfs:
        all_chunks.extend(process_pdf(pdf))
    return all_chunks
