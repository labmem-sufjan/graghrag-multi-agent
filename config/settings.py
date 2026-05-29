"""Central configuration for paths, Neo4j, Chroma, and Ollama."""

from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DOCS_DIR = PROJECT_ROOT / "data" / "raw_docs"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Neo4j (graph only — no vector index)
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "mWsgSmAEv65wb2oqO3NLUag9AzPKh9U-dhP9xPS1acw"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "qwen2.5:7b"
    ollama_embed_model: str = "nomic-embed-text"

    # Chroma
    chroma_collection: str = "prospectus_chunks"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Extraction
    extraction_batch_size: int = 1
    extraction_max_retries: int = 3
    extraction_json_mode: bool = True

    # Retrieval / agents
    retrieval_top_k: int = 5
    graph_entity_limit: int = 8
    graph_hop_chunks: int = 6
    context_max_chunks: int = 10
    context_chunk_max_chars: int = 800

    # Critic: 默认仅用规则（更稳）；设为 true 时再调 LLM
    critic_use_llm: bool = False

    # 发行人 / 文档配置（换 PDF 时优先改 YAML，而非改 agents 代码）
    document_profile_path: str = str(PROJECT_ROOT / "config" / "document_profile.yml")


settings = Settings()
