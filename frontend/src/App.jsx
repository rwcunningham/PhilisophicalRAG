import { useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  BookOpen,
  CheckCircle2,
  Database,
  Loader2,
  RefreshCw,
  RotateCcw,
  Send,
  Server,
  XCircle,
} from "lucide-react";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8787";

function App() {
  const [status, setStatus] = useState(null);
  const [statusError, setStatusError] = useState("");
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [isIndexing, setIsIndexing] = useState(false);
  const scrollRef = useRef(null);

  useEffect(() => {
    refreshStatus();
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  const setupState = useMemo(() => describeSetup(status, statusError), [status, statusError]);

  async function refreshStatus() {
    try {
      setStatusError("");
      const payload = await apiGet("/api/status");
      setStatus(payload);
    } catch (error) {
      setStatus(null);
      setStatusError(error.message);
    }
  }

  async function sendMessage() {
    const message = draft.trim();
    if (!message || isSending) {
      return;
    }

    const history = messages
      .filter((item) => item.role === "user" || item.role === "assistant")
      .filter((item) => item.content && !item.pending && !item.error)
      .map((item) => ({ role: item.role, content: item.content }));

    const userMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: message,
    };
    const pendingId = crypto.randomUUID();
    const pendingMessage = {
      id: pendingId,
      role: "assistant",
      content: "Thinking through the strongest sourced reply...",
      pending: true,
    };

    setDraft("");
    setIsSending(true);
    setMessages((current) => [...current, userMessage, pendingMessage]);

    try {
      const payload = await apiPost("/api/chat", { message, history });
      setStatus(payload.status ?? status);
      setMessages((current) =>
        current.map((item) =>
          item.id === pendingId
            ? {
                id: pendingId,
                role: "assistant",
                content: payload.answer,
                retrievalQuery: payload.retrievalQuery,
                sources: payload.sources ?? [],
                warnings: payload.warnings ?? [],
              }
            : item,
        ),
      );
    } catch (error) {
      setMessages((current) =>
        current.map((item) =>
          item.id === pendingId
            ? {
                id: pendingId,
                role: "assistant",
                content: error.message,
                error: true,
                status: error.status,
              }
            : item,
        ),
      );
      if (error.status) {
        setStatus(error.status);
      }
    } finally {
      setIsSending(false);
    }
  }

  async function buildIndex() {
    if (isIndexing) {
      return;
    }

    setIsIndexing(true);
    const pendingId = crypto.randomUUID();
    setMessages((current) => [
      ...current,
      {
        id: pendingId,
        role: "system",
        content: "Building index...",
        pending: true,
      },
    ]);

    try {
      const payload = await apiPost("/api/index", { reset: true });
      setStatus(payload.status);
      setMessages((current) =>
        current.map((item) =>
          item.id === pendingId
            ? {
                id: pendingId,
                role: "system",
                content: `Indexed ${payload.sourceCount} source document(s) into ${payload.chunkCount} chunks.`,
              }
            : item,
        ),
      );
    } catch (error) {
      setMessages((current) =>
        current.map((item) =>
          item.id === pendingId
            ? {
                id: pendingId,
                role: "system",
                content: error.message,
                error: true,
                status: error.status,
              }
            : item,
        ),
      );
      if (error.status) {
        setStatus(error.status);
      }
    } finally {
      setIsIndexing(false);
    }
  }

  function resetChat() {
    setMessages([]);
    setDraft("");
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-row">
          <BookOpen size={22} aria-hidden="true" />
          <div>
            <h1>Philosophy RAG</h1>
            <p>{setupState.label}</p>
          </div>
        </div>

        <section className="status-panel" aria-label="System status">
          <StatusRow
            icon={<Server size={17} />}
            label="API"
            value={statusError ? "Offline" : "Online"}
            tone={statusError ? "bad" : "good"}
          />
          <StatusRow
            icon={<CheckCircle2 size={17} />}
            label="OpenAI key"
            value={status?.hasOpenAIKey ? "Set" : "Missing"}
            tone={status?.hasOpenAIKey ? "good" : "bad"}
          />
          <StatusRow
            icon={<Database size={17} />}
            label="Vector index"
            value={status?.indexExists ? "Ready" : "Missing"}
            tone={status?.indexExists ? "good" : "warn"}
          />
          <StatusRow
            icon={<BookOpen size={17} />}
            label="Sources"
            value={status ? String(status.sourceCount) : "-"}
            tone={status?.sourceCount > 0 ? "good" : "warn"}
          />
        </section>

        <section className="actions-panel" aria-label="Setup actions">
          <button className="button secondary" onClick={refreshStatus} title="Refresh status">
            <RefreshCw size={17} aria-hidden="true" />
            Refresh
          </button>
          <button
            className="button primary"
            onClick={buildIndex}
            disabled={!status?.canIndex || isIndexing}
            title="Build vector index"
          >
            {isIndexing ? <Loader2 className="spin" size={17} /> : <Database size={17} />}
            {isIndexing ? "Indexing" : "Build index"}
          </button>
        </section>

        {statusError ? <Notice tone="bad" text={statusError} /> : null}
        {status && !status.corpusExists ? <Notice tone="bad" text="Corpus directory not found." /> : null}
        {status && status.corpusExists && status.sourceCount === 0 ? (
          <Notice tone="warn" text="No source texts found in the configured corpus." />
        ) : null}

        {status ? (
          <section className="paths-panel" aria-label="Configuration">
            <PathLine label="Corpus" value={status.paths?.corpus} />
            <PathLine label="Index" value={status.paths?.index} />
            <PathLine label="Chat model" value={status.models?.chat} />
            <PathLine label="Embeddings" value={status.models?.embedding} />
          </section>
        ) : null}

        {status?.sampleSources?.length ? (
          <section className="sources-panel" aria-label="Source sample">
            <h2>Source Sample</h2>
            <ul>
              {status.sampleSources.map((source) => (
                <li key={source}>{source}</li>
              ))}
            </ul>
          </section>
        ) : null}
      </aside>

      <main className="chat-area">
        <header className="chat-header">
          <div>
            <h2>Counterargument Chat</h2>
            <p>{setupState.detail}</p>
          </div>
          <button className="icon-button" onClick={resetChat} title="Clear conversation" aria-label="Clear conversation">
            <RotateCcw size={18} aria-hidden="true" />
          </button>
        </header>

        <div className="messages" ref={scrollRef} aria-live="polite">
          {messages.length === 0 ? <EmptyState status={status} statusError={statusError} /> : null}
          {messages.map((message) => (
            <Message key={message.id} message={message} />
          ))}
        </div>

        <div className="composer">
          <textarea
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
                event.preventDefault();
                sendMessage();
              }
            }}
            placeholder="State a philosophical claim..."
            rows={3}
          />
          <button className="send-button" onClick={sendMessage} disabled={isSending || !draft.trim()} title="Send claim">
            {isSending ? <Loader2 className="spin" size={19} /> : <Send size={19} />}
            Send
          </button>
        </div>
      </main>
    </div>
  );
}

function Message({ message }) {
  const isAssistant = message.role === "assistant";
  return (
    <article className={`message ${message.role} ${message.error ? "error" : ""}`}>
      <div className="message-label">{labelForRole(message.role)}</div>
      <div className="message-body">
        {message.pending ? <Loader2 className="inline-spin spin" size={16} aria-hidden="true" /> : null}
        <div className="answer-text">{message.content}</div>
      </div>

      {isAssistant && message.retrievalQuery ? (
        <div className="retrieval-query">
          <span>Retrieval query</span>
          <code>{message.retrievalQuery}</code>
        </div>
      ) : null}

      {message.warnings?.length ? (
        <div className="warnings">
          {message.warnings.map((warning) => (
            <div className="warning-line" key={warning}>
              <AlertTriangle size={15} aria-hidden="true" />
              <span>{warning}</span>
            </div>
          ))}
        </div>
      ) : null}

      {message.sources?.length ? (
        <div className="source-list">
          {message.sources.map((source) => (
            <details className="source-row" key={source.label}>
              <summary>
                <span className="citation">{source.citation}</span>
                <span>{source.title}</span>
              </summary>
              <p>{source.text}</p>
            </details>
          ))}
        </div>
      ) : null}
    </article>
  );
}

function EmptyState({ status, statusError }) {
  let icon = <CheckCircle2 size={24} aria-hidden="true" />;
  let text = "Ready for a claim.";
  if (statusError) {
    icon = <XCircle size={24} aria-hidden="true" />;
    text = "API unavailable.";
  } else if (!status?.canAsk) {
    icon = <AlertTriangle size={24} aria-hidden="true" />;
    text = "Index required before chat.";
  }

  return (
    <div className="empty-state">
      {icon}
      <p>{text}</p>
    </div>
  );
}

function StatusRow({ icon, label, value, tone }) {
  return (
    <div className={`status-row ${tone}`}>
      <span className="status-icon">{icon}</span>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function PathLine({ label, value }) {
  return (
    <div className="path-line">
      <span>{label}</span>
      <code>{value || "-"}</code>
    </div>
  );
}

function Notice({ tone, text }) {
  return (
    <div className={`notice ${tone}`}>
      {tone === "bad" ? <XCircle size={16} aria-hidden="true" /> : <AlertTriangle size={16} aria-hidden="true" />}
      <span>{text}</span>
    </div>
  );
}

function describeSetup(status, statusError) {
  if (statusError) {
    return {
      label: "API offline",
      detail: "Start the API server, then refresh.",
    };
  }
  if (!status) {
    return {
      label: "Checking",
      detail: "Checking local RAG state.",
    };
  }
  if (!status.hasOpenAIKey) {
    return {
      label: "Needs key",
      detail: "Add OPENAI_API_KEY to the project .env.",
    };
  }
  if (!status.indexExists) {
    return {
      label: "Needs index",
      detail: "Build the vector index before asking claims.",
    };
  }
  return {
    label: "Ready",
    detail: `${status.sourceCount} source document${status.sourceCount === 1 ? "" : "s"} available.`,
  };
}

function labelForRole(role) {
  if (role === "assistant") {
    return "Counterargument";
  }
  if (role === "system") {
    return "System";
  }
  return "You";
}

async function apiGet(path) {
  const response = await fetch(`${API_BASE}${path}`);
  return readApiResponse(response);
}

async function apiPost(path, body) {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return readApiResponse(response);
}

async function readApiResponse(response) {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok || payload.ok === false) {
    const error = new Error(payload.error || `Request failed with ${response.status}`);
    error.status = payload.status;
    throw error;
  }
  return payload;
}

export default App;
