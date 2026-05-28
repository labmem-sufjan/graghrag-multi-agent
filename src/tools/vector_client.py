"""Chroma vector store for document chunks (embeddings live here only)."""

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
    """Embed and store chunks in Chroma. Returns number of chunks indexed."""
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
    store = get_vector_store()
    return store.similarity_search(query, k=k)
