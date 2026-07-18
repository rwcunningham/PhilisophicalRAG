from __future__ import annotations

import argparse
import json
import os
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .config import Settings, get_settings
from .index import build_index, index_exists
from .loaders import SUPPORTED_SUFFIXES
from .rag import PhilosophyRAG, SourcePassage


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787
MAX_BODY_BYTES = 2_000_000

_rag_lock = threading.Lock()
_rag_instance: PhilosophyRAG | None = None
_rag_signature: tuple[str, str, str, str] | None = None


class PhilosophyRAGRequestHandler(BaseHTTPRequestHandler):
    server_version = "PhilosophyRAGAPI/0.1"

    def do_OPTIONS(self) -> None:
        self._send_empty(HTTPStatus.NO_CONTENT)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/health":
            self._send_json({"ok": True})
            return
        if path == "/api/status":
            self._send_json(status_payload())
            return
        self._send_json({"ok": False, "error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/chat":
            self._handle_chat()
            return
        if path == "/api/index":
            self._handle_index()
            return
        self._send_json({"ok": False, "error": "Not found"}, HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")

    def _handle_chat(self) -> None:
        try:
            payload = self._read_json()
            message = str(payload.get("message", "")).strip()
            if not message:
                self._send_json({"ok": False, "error": "Message is required."}, HTTPStatus.BAD_REQUEST)
                return

            history = _history_from_payload(payload.get("history", []))
            settings = get_settings()
            with _rag_lock:
                rag = _get_rag(settings)
                response = rag.answer(message, history=history)

            self._send_json(
                {
                    "ok": True,
                    "answer": response.answer,
                    "retrievalQuery": response.retrieval_query,
                    "warnings": response.warnings,
                    "sources": [_source_payload(source) for source in response.sources],
                    "status": status_payload(settings),
                }
            )
        except Exception as exc:
            self._send_error_payload(exc)

    def _handle_index(self) -> None:
        try:
            payload = self._read_json()
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            reset = _coerce_bool(payload.get("reset", query.get("reset", [True])[0]), default=True)
            settings = get_settings()
            source_count, chunk_count = build_index(settings, reset=reset)
            _reset_rag_cache()
            self._send_json(
                {
                    "ok": True,
                    "sourceCount": source_count,
                    "chunkCount": chunk_count,
                    "status": status_payload(settings),
                }
            )
        except Exception as exc:
            self._send_error_payload(exc)

    def _read_json(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError("Invalid Content-Length header.") from exc
        if length > MAX_BODY_BYTES:
            raise ValueError("Request body is too large.")
        if length == 0:
            return {}

        raw = self.rfile.read(length).decode("utf-8")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("Request body must be valid JSON.") from exc
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")
        return payload

    def _send_error_payload(self, exc: Exception) -> None:
        code = HTTPStatus.BAD_REQUEST if isinstance(exc, ValueError) else HTTPStatus.CONFLICT
        self._send_json(
            {
                "ok": False,
                "error": str(exc),
                "status": status_payload(),
            },
            code,
        )

    def _send_empty(self, status: HTTPStatus) -> None:
        self.send_response(status)
        self._send_cors_headers()
        self.end_headers()

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")


class PhilosophyRAGServer(ThreadingHTTPServer):
    daemon_threads = True


def status_payload(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    source_files = _source_files(settings.corpus_dir)
    has_index = index_exists(settings.index_dir)
    has_key = settings.has_openai_key

    return {
        "ok": True,
        "hasOpenAIKey": has_key,
        "corpusExists": settings.corpus_dir.exists(),
        "indexExists": has_index,
        "canAsk": has_key and has_index,
        "canIndex": has_key and bool(source_files),
        "sourceCount": len(source_files),
        "sampleSources": [_display_path(path, settings.corpus_dir) for path in source_files[:8]],
        "paths": {
            "corpus": str(settings.corpus_dir),
            "index": str(settings.index_dir),
        },
        "models": {
            "chat": settings.chat_model,
            "embedding": settings.embedding_model,
        },
    }


def run_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    server = PhilosophyRAGServer((host, port), PhilosophyRAGRequestHandler)
    print(f"Philosophy RAG API listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Philosophy RAG API.")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Philosophy RAG JSON API.")
    parser.add_argument("--host", default=os.getenv("PHILOSOPHY_RAG_API_HOST", DEFAULT_HOST))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PHILOSOPHY_RAG_API_PORT", str(DEFAULT_PORT))),
    )
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)


def _get_rag(settings: Settings) -> PhilosophyRAG:
    global _rag_instance, _rag_signature

    signature = (
        str(settings.index_dir),
        settings.collection_name,
        settings.chat_model,
        settings.embedding_model,
    )
    if _rag_instance is None or _rag_signature != signature:
        _rag_instance = PhilosophyRAG(settings)
        _rag_signature = signature
    return _rag_instance


def _reset_rag_cache() -> None:
    global _rag_instance, _rag_signature
    with _rag_lock:
        _rag_instance = None
        _rag_signature = None


def _history_from_payload(payload: Any) -> list[tuple[str, str]]:
    if not isinstance(payload, list):
        return []

    history: list[tuple[str, str]] = []
    for item in payload[-16:]:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        role = str(item.get("role", "user")).lower()
        speaker = "Assistant" if role == "assistant" else "User"
        history.append((speaker, content))
    return history


def _source_payload(source: SourcePassage) -> dict[str, Any]:
    return {
        "label": source.label,
        "citation": source.citation,
        "title": _source_title(source.metadata),
        "text": source.text,
        "metadata": source.metadata,
    }


def _source_title(metadata: dict[str, Any]) -> str:
    parts = []
    for key in ("author", "title", "work", "source", "page"):
        value = metadata.get(key)
        if value not in (None, ""):
            parts.append(str(value))
    return " | ".join(parts) or "Unknown source"


def _source_files(corpus_dir: Path) -> list[Path]:
    if not corpus_dir.exists():
        return []

    files = []
    for path in sorted(corpus_dir.rglob("*")):
        if not path.is_file() or _should_skip_source_file(path):
            continue
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(path)
    return files


def _should_skip_source_file(path: Path) -> bool:
    name = path.name.lower()
    return name.startswith(".") or name == "readme.md" or name.endswith(".metadata.json")


def _display_path(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


if __name__ == "__main__":
    main()
