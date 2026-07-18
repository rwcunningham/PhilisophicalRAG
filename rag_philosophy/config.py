from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _path_from_env(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    path = Path(raw) if raw else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.expanduser().resolve()


def _int_from_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


def _float_from_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw else default


def _bool_from_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    corpus_dir: Path = _path_from_env("PHILOSOPHY_RAG_CORPUS", PROJECT_ROOT / "data" / "texts")
    index_dir: Path = _path_from_env("PHILOSOPHY_RAG_INDEX", PROJECT_ROOT / "data" / "index" / "chroma")
    collection_name: str = os.getenv("PHILOSOPHY_RAG_COLLECTION", "philosophy_sources")

    chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-5-mini")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

    chunk_size: int = _int_from_env("PHILOSOPHY_RAG_CHUNK_SIZE", 1200)
    chunk_overlap: int = _int_from_env("PHILOSOPHY_RAG_CHUNK_OVERLAP", 180)

    retriever_k: int = _int_from_env("PHILOSOPHY_RAG_RETRIEVER_K", 12)
    retriever_fetch_k: int = _int_from_env("PHILOSOPHY_RAG_RETRIEVER_FETCH_K", 40)
    final_context_k: int = _int_from_env("PHILOSOPHY_RAG_FINAL_CONTEXT_K", 5)
    mmr_lambda: float = _float_from_env("PHILOSOPHY_RAG_MMR_LAMBDA", 0.45)

    enable_llm_rerank: bool = _bool_from_env("PHILOSOPHY_RAG_ENABLE_LLM_RERANK", True)
    max_context_chars_per_source: int = _int_from_env("PHILOSOPHY_RAG_MAX_CONTEXT_CHARS", 1600)

    @property
    def has_openai_key(self) -> bool:
        key = os.getenv("OPENAI_API_KEY", "").strip()
        return bool(key and key != "sk-your-key-here")


def get_settings() -> Settings:
    return Settings()
