from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from pypdf import PdfReader


SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".pdf"}
FRONTMATTER_RE = re.compile(r"\A---\s*\n(?P<meta>.*?)\n---\s*\n(?P<body>.*)\Z", re.DOTALL)


def load_corpus(corpus_dir: Path) -> list[Document]:
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory does not exist: {corpus_dir}")

    documents: list[Document] = []
    for path in sorted(corpus_dir.rglob("*")):
        if not path.is_file() or _should_skip_file(path):
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        documents.extend(load_source_file(path, corpus_dir))

    return documents


def load_source_file(path: Path, corpus_dir: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path, corpus_dir)
    return [_load_text(path, corpus_dir)]


def _load_text(path: Path, corpus_dir: Path) -> Document:
    text = path.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(text)
    metadata = _base_metadata(path, corpus_dir)
    metadata.update(frontmatter)
    metadata.update(_sidecar_metadata(path))

    return Document(page_content=body.strip(), metadata=metadata)


def _load_pdf(path: Path, corpus_dir: Path) -> list[Document]:
    metadata = _base_metadata(path, corpus_dir)
    metadata.update(_sidecar_metadata(path))

    reader = PdfReader(str(path))
    documents: list[Document] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue
        page_metadata = dict(metadata)
        page_metadata["page"] = index
        documents.append(Document(page_content=text.strip(), metadata=page_metadata))
    return documents


def _base_metadata(path: Path, corpus_dir: Path) -> dict[str, Any]:
    try:
        source = str(path.relative_to(corpus_dir))
    except ValueError:
        source = str(path)

    return {
        "source": source,
        "title": path.stem.replace("_", " ").replace("-", " ").title(),
        "file_name": path.name,
        "file_type": path.suffix.lower().lstrip("."),
    }


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text

    metadata: dict[str, Any] = {}
    for line in match.group("meta").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if "," in value and key in {"tags", "topics", "keywords"}:
            metadata[key] = [item.strip() for item in value.split(",") if item.strip()]
        else:
            metadata[key] = value

    return metadata, match.group("body")


def _sidecar_metadata(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(".metadata.json")
    if not sidecar.exists():
        return {}
    with sidecar.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Sidecar metadata must be a JSON object: {sidecar}")
    return payload


def _should_skip_file(path: Path) -> bool:
    name = path.name.lower()
    return (
        name.startswith(".")
        or name == "readme.md"
        or name.endswith(".metadata.json")
    )
