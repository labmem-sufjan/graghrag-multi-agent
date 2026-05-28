"""
本模块负责 RAG 系统离线流水线（Data Pipeline）的前半部分：
功能包括：读取指定目录下的 PDF 招股书、文本级深度清洗、动态章节（目录）感知追踪、
以及利用 LangChain 切片器进行防断句的语义级文本切片，最终输出带有工业级溯源元数据的 Document 集合。
"""

from __future__ import annotations

import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 引入项目全局配置参数（如默认的 chunk_size, chunk_overlap 以及默认数据目录）
from config.settings import RAW_DOCS_DIR, settings

# ==========================================
#    正则表达式预编译区（提升高频文本匹配效率）
# ==========================================

# 1. 章节标题匹配正则： 用于捕获中国标准招股书或财报中的章节符号
# 匹配规则：以“第一章”、“第1节”、“一、”或“1.”开头的行
_SECTION_PATTERN = re.compile(
    r"^(第[一二三四五六七八九十百零\d]+[章节篇部]|[一二三四五六七八九十]+[、．.])"
)

# 2. 空白符清洗正则：匹配连续的空格、制表符（\t）或全角空格（\u3000）
_WS_PATTERN = re.compile(r"[ \t\u3000]+")

# 3. 极端换行符清洗正则：匹配连续 3 个及以上的换行符
_MULTI_NL = re.compile(r"\n{3,}")


def _clean_text(text: str) -> str:
    """
    【文本层深度清洗函数】
    目的：去除 PDF 解析产生的排版噪声，防止污染 Embedding 向量表示，提高大模型图谱抽取率。
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")  # 统一换行符
    text = _WS_PATTERN.sub(" ", text)                                    # 把连续的空格/制表符缩减为一个空格
    text = _MULTI_NL.sub("\n\n", text)                                   # 把多于3个的连续换行符缩减为双换行
    return text.strip()


def _detect_section(line: str, current: str | None) -> str | None:
    """
        【动态章节标题感知函数】
        机制：基于状态机（State Machine），扫描当前行是否为新的章节大标题。
        """
    line = line.strip()
    if len(line) > 80:                # 长度超过80肯定不是标题，直接跳过
        return current
    if _SECTION_PATTERN.match(line):  # 匹配类似 “第二章”、“一、” 开头的行
        return line[:120]             # 截取前120字作为全新的当前章节名返回
    return current                    # 没发现新标题，沿用旧标题


def load_pdf(path: Path) -> list[Document]:
    """
    【PDF 文件加载器】
    功能：将单个 PDF 读入内存，按照“一页 = 一个 Document”的粒度切分，并进行初步文本清洗与元数据注入。
    """
    loader = PyPDFLoader(str(path))  # 初始化 LangChain 的 PDF 异步/同步加载器
    pages = loader.load()            # 执行加载：此时 pages 是一个 List[Document]，长度等于 PDF 总页数
    doc_name = path.name
    # 遍历每一页，注入原始文件的追踪元数据（Provenance Metadata）
    for page in pages:
        page.metadata["source"] = doc_name
        page.metadata["source_path"] = str(path)
        # 对当前页的原始正文进行正则噪音清洗
        page.page_content = _clean_text(page.page_content)
    return pages


def chunk_documents(
    pages: list[Document],
    *,
    doc_stem: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """
    【核心：高级切片与元数据富化函数】
    功能：将按页划分的原始文档拆分为固定长度且有语义重叠的较小 Chunk，并生成工业级可追溯的 chunk_id。
    """
    if not pages:
        return []

    # 获取基础文件名用于拼接唯一的 chunk_id
    source_name = pages[0].metadata.get("source", "unknown.pdf")
    stem = doc_stem or Path(source_name).stem

    # 初始化递归字符文本切片器
    # 机制：它会按照 separators 列表的顺序由高到低尝试切分。
    # 优先在段落（\n\n）切，太长就在换行（\n）切，再长就在句号（。）切，最大程度确保一句话或一段逻辑的完整性。
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        separators=["\n\n", "\n", "。", "；", " ", ""],
    )

    chunks: list[Document] = []
    section: str | None = None  # 章节状态机变量：记录当前切片属于招股书的哪一章
    global_idx = 0              # 文档全局切片自增索引

    for page in pages:
        page_num = page.metadata.get("page", 0)                                      # PyPDFLoader 默认生成的 page 元数据是从 0 开始计数的
        display_page = int(page_num) + 1 if isinstance(page_num, int) else page_num  # 将页码从第 1 页开始计

        # 【关键步骤 1】：在对整页进行切片前，先逐行扫描该页，更新当前正文所处的章节目录（Section）
        for line in page.page_content.split("\n"):
            section = _detect_section(line, section)

        # 【关键步骤 2】：调用 LangChain 切片器，将本页正文肢解为多个正文碎片（pieces）
        for piece in splitter.split_text(page.page_content):
            piece = piece.strip()
            if not piece:
                continue
            # 【关键步骤 3】：拼装工业级规范的全局唯一稳定 ID（Stable Chunk ID）
            chunk_id = f"{stem}_p{display_page:03d}_c{global_idx:04d}"  # 为每一个 Document 对象定义规范的 metadata
            global_idx += 1
            # 封装并重新构建一个标准化、高元数据密度的新 Document 对象
            chunks.append(
                Document(
                    page_content=piece,
                    metadata={
                        "source": source_name,                                # 原始文件名
                        "source_path": page.metadata.get("source_path", ""),  # 文件磁盘绝对路径
                        "page": display_page,                                 # 纠正后的真实页码
                        "chunk_id": chunk_id,                                 # 全局唯一块 id
                        "section": section or "",                             # 动态感知到的所属章节标题
                        "document_stem": stem,                                # 不带后缀的文件名
                    },
                )
            )
    return chunks


def process_pdf(path: Path) -> list[Document]:
    """
    【单文件处理微工作流】
    串联单个 PDF 文件的“检查、加载、切片”完整闭环。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    # 1. 物理加载与清洗页
    pages = load_pdf(path)
    # 2. 深入切片并打上丰富标签，返回最终可直接喂给 Chroma 和 Neo4j 的 chunk 列表
    return chunk_documents(pages, doc_stem=path.stem)


def process_directory(directory: Path | None = None) -> list[Document]:
    """
    【批处理批操作入口函数】
    功能：遍历指定目录下所有的 PDF 文件，进行批量的自动化流水线解析。
    """
    directory = Path(directory or RAW_DOCS_DIR)  # 如果未传入路径，默认去全局配置项指向的原始文档存放目录（如 ./data/raw_docs）读取
    pdfs = sorted(directory.glob("*.pdf"))       # 排序检索目录下所有以 .pdf 结尾的文件
    if not pdfs:
        raise FileNotFoundError(f"No PDF files in {directory}")

    all_chunks: list[Document] = []
    # 循环调用单文件流水线，将所有 PDF 产出的 chunk 汇聚到一个大列表里
    for pdf in pdfs:
        all_chunks.extend(process_pdf(pdf))
    return all_chunks
