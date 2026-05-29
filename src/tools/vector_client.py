"""Chroma 向量库：仅存 chunk 文本与 embedding，不负责图谱。

离线 index_chunks 由 build_knowledge 调用；在线 similarity_search 由 vector/hybrid expert 调用。
"""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from config.settings import CHROMA_DIR, settings


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=settings.ollama_embed_model,
        base_url=settings.ollama_base_url,
    )


def get_vector_store() -> Chroma:
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def index_chunks(chunks: list[Document], *, reset: bool = False) -> int:
    """写入 Chroma；id 使用 chunk_id，便于去重与评测对齐。"""
    if not chunks:
        return 0

    ids = [c.metadata["chunk_id"] for c in chunks]
    if reset:
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            client.delete_collection(settings.chroma_collection)
        except Exception:
            pass

    store = get_vector_store()
    store.add_documents(chunks, ids=ids)
    return len(chunks)


def similarity_search(query: str, k: int = 5) -> list[Document]:
    """在线语义检索，返回 Top-K Document（含 metadata）。"""
    store = get_vector_store()
    return store.similarity_search(query, k=k)
