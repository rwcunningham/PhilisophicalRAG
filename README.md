# Philosophy RAG

A local Python RAG system for generating sourced philosophical counterarguments.

The intended workflow is:

1. Put real philosophical texts in `data/texts/`.
2. Build a persistent vector index with OpenAI embeddings and Chroma.
3. Ask for a counterargument.
4. Continue the argument in an interactive chat loop; each reply retrieves fresh passages and cites them.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Paste your OpenAI API key into `.env`:

```bash
OPENAI_API_KEY=sk-...
```

Keep `.env` private. It is already ignored by Git.

## Add source texts

Place `.txt`, `.md`, or `.pdf` files in `data/texts/`. Markdown and text files may include optional frontmatter:

```markdown
---
title: Ethics
author: Baruch Spinoza
work: Ethics
year: 1677
tags: determinism, free will, necessity
---

Text begins here...
```

PDFs are indexed page by page. Text and Markdown files are indexed as documents, then chunked.

## Build the index

```bash
philosophy-rag index
```

This creates a persistent Chroma index under `data/index/chroma`.

## Ask for a counterargument

```bash
philosophy-rag ask "man has free will"
```

The system rewrites the claim into an adversarial retrieval query, performs MMR retrieval, optionally reranks passages with the chat model, then generates a citation-grounded answer.

## Continue arguing

```bash
philosophy-rag chat
```

Each turn retrieves fresh source passages for the newest user claim while preserving the recent debate context.

## Run the React frontend

Start the local JSON API:

```bash
philosophy-rag-api
```

In another terminal, start the React app:

```bash
cd frontend
npm install
npm run dev
```

The frontend expects the API at `http://127.0.0.1:8787`. To use a different API URL, create `frontend/.env.local` with:

```bash
VITE_API_BASE=http://127.0.0.1:8787
```

## Useful commands

```bash
philosophy-rag sources
philosophy-rag index --no-reset
philosophy-rag ask "morality requires God" --show-context
```

## What makes this a full RAG pipeline

- Corpus ingestion for `.txt`, `.md`, and `.pdf`.
- Source metadata extraction from frontmatter and sidecar JSON.
- Recursive semantic chunking with stable chunk IDs.
- OpenAI embeddings.
- Persistent Chroma vector store.
- Query rewriting for counterargument retrieval.
- Max marginal relevance retrieval for diversity.
- LLM reranking for philosophical relevance.
- Grounded answer generation with citation IDs.
- Citation validation warnings when an answer is not properly sourced.
