from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings
from .loaders import load_corpus


def make_embeddings(settings: Settings) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=settings.embedding_model)


def make_vector_store(settings: Settings) -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=make_embeddings(settings),
        persist_directory=str(settings.index_dir),
    )


def split_documents(documents: list[Document], settings: Settings) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "; ",
            ", ",
            " ",
            "",
        ],
    )
    chunks = splitter.split_documents([doc for doc in documents if doc.page_content.strip()])

    for chunk_number, chunk in enumerate(chunks, start=1):
        chunk.metadata = _normalize_metadata(chunk.metadata)
        chunk.metadata["chunk_number"] = chunk_number
        chunk.metadata["chunk_id"] = stable_chunk_id(chunk)

    return chunks


def build_index(settings: Settings, reset: bool = True) -> tuple[int, int]:
    if not settings.has_openai_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Copy .env.example to .env and paste your key there.")

    source_documents = load_corpus(settings.corpus_dir)
    chunks = split_documents(source_documents, settings)
    if not chunks:
        raise RuntimeError(f"No indexable text found in {settings.corpus_dir}")

    if reset and settings.index_dir.exists():
        shutil.rmtree(settings.index_dir)

    settings.index_dir.mkdir(parents=True, exist_ok=True)
    vector_store = make_vector_store(settings)
    ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    vector_store.add_documents(chunks, ids=ids)

    return len(source_documents), len(chunks)


def stable_chunk_id(document: Document) -> str:
    metadata = document.metadata
    seed = "|".join(
        [
            str(metadata.get("source", "")),
            str(metadata.get("page", "")),
            str(metadata.get("start_index", "")),
            document.page_content[:160],
        ]
    )
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()


def _normalize_metadata(metadata: dict) -> dict:
    normalized = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            normalized[key] = value
        elif isinstance(value, list):
            normalized[key] = ", ".join(str(item) for item in value)
        else:
            normalized[key] = str(value)
    return normalized


def index_exists(index_dir: Path) -> bool:
    return index_dir.exists() and any(index_dir.iterdir())
