# Philosophy RAG Agent Notes

## Startup Runbook

Use the repo root:

```bash
cd "/Users/robertcunningham/Desktop/CU Boulder 2025/Philosophy RAG"
```

Start the local RAG API in one terminal:

```bash
source .venv/bin/activate
philosophy-rag-api --host 127.0.0.1 --port 8787
```

Start the React frontend in another terminal:

```bash
cd "/Users/robertcunningham/Desktop/CU Boulder 2025/Philosophy RAG/frontend"
npm run dev
```

Open the app at:

```text
http://127.0.0.1:5173/
```

The frontend expects the API at:

```text
http://127.0.0.1:8787
```

## Required Local State

- The OpenAI key belongs in the repo-root `.env` file as `OPENAI_API_KEY=...`.
- Source texts belong in `data/texts/`.
- The Chroma index is stored at `data/index/chroma`.

If the app says the index is missing, rebuild it from the repo root:

```bash
source .venv/bin/activate
philosophy-rag sources
philosophy-rag index
```

## Quick Checks

Check the API:

```bash
curl -s http://127.0.0.1:8787/api/status
```

Healthy status should include:

```json
{
  "hasOpenAIKey": true,
  "indexExists": true,
  "canAsk": true
}
```

Check the frontend:

```bash
curl -I http://127.0.0.1:5173/
```

Expected response is `HTTP/1.1 200 OK`.

## Notes

- `philosophy-rag-api` serves `/api/status`, `/api/index`, and `/api/chat`.
- `/api/chat` uses the configured OpenAI key, the local Chroma index, and the existing `PhilosophyRAG.answer()` flow.
- If `philosophy-rag` or `philosophy-rag-api` points at an old checkout, refresh the editable install from this repo root:

```bash
source .venv/bin/activate
python -m pip install -e .
```
