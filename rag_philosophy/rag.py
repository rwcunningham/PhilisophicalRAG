from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from .config import Settings
from .index import index_exists, make_vector_store
from .prompts import ANSWER_PROMPT, QUERY_REWRITE_PROMPT, RERANK_PROMPT


@dataclass(frozen=True)
class SourcePassage:
    label: str
    text: str
    metadata: dict

    @property
    def citation(self) -> str:
        return f"[{self.label}]"


@dataclass(frozen=True)
class RAGResponse:
    answer: str
    retrieval_query: str
    sources: list[SourcePassage]
    warnings: list[str]


class PhilosophyRAG:
    def __init__(self, settings: Settings):
        if not settings.has_openai_key:
            raise RuntimeError("OPENAI_API_KEY is missing. Copy .env.example to .env and paste your key there.")
        if not index_exists(settings.index_dir):
            raise RuntimeError(f"No vector index found at {settings.index_dir}. Run `philosophy-rag index` first.")

        self.settings = settings
        self.llm = ChatOpenAI(model=settings.chat_model)
        self.vector_store = make_vector_store(settings)

    def answer(self, claim: str, history: list[tuple[str, str]] | None = None) -> RAGResponse:
        history = history or []
        retrieval_query = self.rewrite_query(claim)
        candidates = self.retrieve(retrieval_query)
        selected = self.rerank(claim, candidates) if self.settings.enable_llm_rerank else candidates[: self.settings.final_context_k]
        sources = self._to_source_passages(selected)

        response = self.llm.invoke(
            ANSWER_PROMPT.format_messages(
                history=_format_history(history),
                claim=claim,
                context=_format_context(sources),
            )
        )
        answer = str(response.content).strip()
        warnings = validate_citations(answer, sources)
        return RAGResponse(answer=answer, retrieval_query=retrieval_query, sources=sources, warnings=warnings)

    def rewrite_query(self, claim: str) -> str:
        response = self.llm.invoke(QUERY_REWRITE_PROMPT.format_messages(claim=claim))
        query = str(response.content).strip().strip('"')
        return query or claim

    def retrieve(self, retrieval_query: str) -> list[Document]:
        return self.vector_store.max_marginal_relevance_search(
            retrieval_query,
            k=self.settings.retriever_k,
            fetch_k=self.settings.retriever_fetch_k,
            lambda_mult=self.settings.mmr_lambda,
        )

    def rerank(self, claim: str, documents: list[Document]) -> list[Document]:
        if len(documents) <= self.settings.final_context_k:
            return documents

        numbered = "\n\n".join(
            f"{index}. {_source_title(doc)}\n{_clip(doc.page_content, 900)}"
            for index, doc in enumerate(documents, start=1)
        )
        response = self.llm.invoke(
            RERANK_PROMPT.format_messages(
                claim=claim,
                passages=numbered,
                limit=self.settings.final_context_k,
            )
        )

        selected_indices = _parse_selected_indices(str(response.content), len(documents))
        if not selected_indices:
            return documents[: self.settings.final_context_k]

        selected: list[Document] = []
        for index in selected_indices:
            document = documents[index - 1]
            if document not in selected:
                selected.append(document)
            if len(selected) >= self.settings.final_context_k:
                break

        return selected or documents[: self.settings.final_context_k]

    def _to_source_passages(self, documents: Iterable[Document]) -> list[SourcePassage]:
        sources: list[SourcePassage] = []
        for index, document in enumerate(documents, start=1):
            sources.append(
                SourcePassage(
                    label=f"S{index}",
                    text=_clip(document.page_content, self.settings.max_context_chars_per_source),
                    metadata=document.metadata,
                )
            )
        return sources


def validate_citations(answer: str, sources: list[SourcePassage]) -> list[str]:
    warnings: list[str] = []
    valid = {source.label for source in sources}
    cited = set(re.findall(r"\[(S\d+)\]", answer))

    if sources and not cited:
        warnings.append("The generated answer did not include source citations.")

    invalid = cited - valid
    if invalid:
        warnings.append(f"The generated answer cited unknown source labels: {', '.join(sorted(invalid))}.")

    unused = valid - cited
    if unused:
        warnings.append(f"Retrieved but unused source labels: {', '.join(sorted(unused))}.")

    return warnings


def _parse_selected_indices(payload: str, max_index: int) -> list[int]:
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", payload, re.DOTALL)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

    selected = parsed.get("selected", []) if isinstance(parsed, dict) else []
    indices: list[int] = []
    for item in selected:
        number = item.get("number") if isinstance(item, dict) else item
        try:
            index = int(number)
        except (TypeError, ValueError):
            continue
        if 1 <= index <= max_index:
            indices.append(index)
    return indices


def _format_history(history: list[tuple[str, str]]) -> str:
    if not history:
        return "No prior turns."
    turns = []
    for speaker, message in history[-8:]:
        turns.append(f"{speaker}: {message}")
    return "\n".join(turns)


def _format_context(sources: list[SourcePassage]) -> str:
    if not sources:
        return "No source passages retrieved."

    return "\n\n".join(
        f"[{source.label}] {_format_source_metadata(source.metadata)}\n{source.text}"
        for source in sources
    )


def _format_source_metadata(metadata: dict) -> str:
    parts = []
    for key in ("author", "title", "work", "year", "source", "page"):
        value = metadata.get(key)
        if value not in (None, ""):
            parts.append(f"{key}: {value}")
    return "; ".join(parts)


def _source_title(document: Document) -> str:
    return _format_source_metadata(document.metadata) or str(document.metadata.get("source", "Unknown source"))


def _clip(text: str, max_chars: int) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 1].rstrip() + "..."
