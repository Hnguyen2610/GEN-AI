"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import {
  createChatSession,
  deleteChatSession,
  getAuthToken,
  getChatSession,
  listChatSessions,
  type ChatMessage,
  type ChatSession,
} from "../../../lib/api/client";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

const MODELS = [
  { id: "llama3.2:1b", name: "Llama 3.2 (1B)" },
  { id: "gemma", name: "Gemma Default" },
  { id: "gemini-2.5-flash", name: "Gemini 2.5 Flash" },
];

type ChatMeta = Record<string, any> | null | undefined;

function normalizeModelId(modelId: string): string {
  if (
    modelId === "gemini-1.5-flash" ||
    modelId === "gemini-1.5-flash-latest"
  ) {
    return "gemini-2.5-flash";
  }

  return modelId;
}

function getDisplayModelName(modelId?: string | null): string {
  const normalized = normalizeModelId(modelId ?? "");
  const matched = MODELS.find((model) => model.id === normalized);
  if (matched) {
    return matched.name;
  }
  if (normalized === "greeting-fast-path") {
    return "Built-in Fast Path";
  }
  if (!normalized) {
    return "Unknown Model";
  }
  return normalized;
}

function getResponseModelName(message: ChatMessage | null, meta: ChatMeta): string {
  const providerModel = meta?.provider?.model;
  if (providerModel) {
    return getDisplayModelName(providerModel);
  }
  return getDisplayModelName(message?.model_name);
}

function getErrorDetail(meta: ChatMeta): string | null {
  if (!meta) {
    return null;
  }
  return meta.error_detail ?? meta.error ?? null;
}

function formatConfidence(value: unknown): string {
  if (typeof value !== "number") {
    return "-";
  }
  return `${Math.round(value * 100)}%`;
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  if (typeof value === "string") {
    return value;
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function hasProcessContent(meta: ChatMeta): boolean {
  return Boolean(
    meta &&
      (
        meta.route ||
        meta.provider ||
        meta.route_reason ||
        meta.route_confidence !== undefined ||
        meta.sql_used ||
        meta.error ||
        meta.error_detail ||
        meta.agent_traces?.length ||
        meta.verification ||
        meta.data_preview?.length
      ),
  );
}

function renderCitations(meta: ChatMeta) {
  if (!meta?.citations?.length) {
    return null;
  }

  return (
    <div className="chatjvb-citations">
      {meta.citations.map((citation: any, index: number) => (
        <div
          key={`${citation.chunk_id ?? citation.original_filename ?? "citation"}-${index}`}
          className="chatjvb-badge"
          title={citation.quote}
        >
          {"DOC "}
          {citation.original_filename}
          {citation.source_page ? ` (p.${citation.source_page})` : ""}
        </div>
      ))}
    </div>
  );
}

function ProcessPanel({
  message,
  meta,
}: {
  message: ChatMessage | null;
  meta: ChatMeta;
}) {
  if (!hasProcessContent(meta)) {
    return null;
  }

  const modelName = getResponseModelName(message, meta);
  const errorDetail = getErrorDetail(meta);
  const route = meta?.route ?? "-";
  const providerName = meta?.provider?.name ?? "-";

  return (
    <details className="chatjvb-process" open={Boolean(errorDetail)}>
      <summary className="chatjvb-process-summary">Process</summary>
      <div className="chatjvb-process-body">
        <div className="chatjvb-process-grid">
          <div className="chatjvb-process-row">
            <span className="chatjvb-process-label">Model</span>
            <span className="chatjvb-process-value">{modelName}</span>
          </div>
          <div className="chatjvb-process-row">
            <span className="chatjvb-process-label">Provider</span>
            <span className="chatjvb-process-value">{providerName}</span>
          </div>
          <div className="chatjvb-process-row">
            <span className="chatjvb-process-label">Route</span>
            <span className="chatjvb-process-value">{formatValue(route)}</span>
          </div>
          <div className="chatjvb-process-row">
            <span className="chatjvb-process-label">Reason</span>
            <span className="chatjvb-process-value">
              {formatValue(meta?.route_reason)}
            </span>
          </div>
          <div className="chatjvb-process-row">
            <span className="chatjvb-process-label">Confidence</span>
            <span className="chatjvb-process-value">
              {formatConfidence(meta?.route_confidence)}
            </span>
          </div>
          <div className="chatjvb-process-row">
            <span className="chatjvb-process-label">Retrieval</span>
            <span className="chatjvb-process-value">
              {formatValue(meta?.retrieval_used ?? message?.retrieval_used)}
            </span>
          </div>
          <div className="chatjvb-process-row">
            <span className="chatjvb-process-label">Rows</span>
            <span className="chatjvb-process-value">
              {formatValue(meta?.row_count)}
            </span>
          </div>
          <div className="chatjvb-process-row">
            <span className="chatjvb-process-label">Error Stage</span>
            <span className="chatjvb-process-value">
              {formatValue(meta?.error_stage)}
            </span>
          </div>
        </div>

        {meta?.sql_used ? (
          <div className="chatjvb-process-section">
            <div className="chatjvb-process-label">SQL</div>
            <pre className="chatjvb-process-code">{meta.sql_used}</pre>
          </div>
        ) : null}

        {Array.isArray(meta?.agent_traces) && meta.agent_traces.length > 0 ? (
          <div className="chatjvb-process-section">
            <div className="chatjvb-process-label">Agent Steps</div>
            <div className="chatjvb-process-traces">
              {meta.agent_traces.map((trace: any, index: number) => (
                <div key={`${trace.tool ?? "tool"}-${index}`} className="chatjvb-process-trace">
                  <strong>Step {trace.step ?? index + 1}</strong>
                  {`: ${trace.tool ?? "unknown"} (${trace.result ?? "unknown"})`}
                  {trace.args ? (
                    <pre className="chatjvb-process-code small">
                      {JSON.stringify(trace.args, null, 2)}
                    </pre>
                  ) : null}
                </div>
              ))}
            </div>
          </div>
        ) : null}

        {meta?.verification ? (
          <div className="chatjvb-process-section">
            <div className="chatjvb-process-label">Verification</div>
            <pre className="chatjvb-process-code small">
              {JSON.stringify(meta.verification, null, 2)}
            </pre>
          </div>
        ) : null}

        {errorDetail ? (
          <div className="chatjvb-process-error">
            <div className="chatjvb-process-label">Error Detail</div>
            <div className="chatjvb-process-error-text">{errorDetail}</div>
          </div>
        ) : null}
      </div>
    </details>
  );
}

function AssistantResponse({
  message,
  meta,
  content,
}: {
  message: ChatMessage | null;
  meta: ChatMeta;
  content: string;
}) {
  const modelName = getResponseModelName(message, meta);
  const errorDetail = getErrorDetail(meta);

  return (
    <div className="chatjvb-assistant-msg">
      <div className="chatjvb-avatar ai">
        <svg
          width="18"
          height="18"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
          <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
          <line x1="12" y1="22.08" x2="12" y2="12"></line>
        </svg>
      </div>
      <div style={{ flex: 1 }}>
        <div>{content}</div>
        <div className="chatjvb-model-meta">Model: {modelName}</div>
        {errorDetail ? (
          <div className="chatjvb-inline-error">{errorDetail}</div>
        ) : null}
        {renderCitations(meta)}
        <ProcessPanel message={message} meta={meta} />
      </div>
    </div>
  );
}

export default function ChatView() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSession, setCurrentSession] = useState<ChatSession | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [model, setModel] = useState(() => normalizeModelId(MODELS[0].id));
  const [loading, setLoading] = useState(false);
  const [deletingSessionId, setDeletingSessionId] = useState<string | null>(null);
  const [streamingToken, setStreamingToken] = useState("");
  const [streamingMeta, setStreamingMeta] = useState<ChatMeta>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, streamingToken, streamingMeta]);

  useEffect(() => {
    void loadSessions();
  }, []);

  useEffect(() => {
    setModel((current) => normalizeModelId(current));
  }, []);

  async function loadSessions() {
    try {
      const data = await listChatSessions();
      setSessions(data);
    } catch (error) {
      console.error("Failed to load sessions", error);
    }
  }

  async function handleCreateSession() {
    const title = prompt("New chat title:");
    if (!title) {
      return;
    }

    try {
      const session = await createChatSession(title);
      setSessions((current) => [session, ...current]);
      await selectSession(session.id);
    } catch {
      alert("Failed to create chat session.");
    }
  }

  async function selectSession(id: string) {
    try {
      const session = await getChatSession(id);
      setCurrentSession(session);
      setMessages(session.messages || []);
      setStreamingToken("");
      setStreamingMeta(null);
    } catch {
      alert("Failed to load chat session.");
    }
  }

  async function handleDeleteSession(sessionId: string) {
    const targetSession = sessions.find((session) => session.id === sessionId);
    if (!targetSession) {
      return;
    }

    const confirmed = window.confirm(
      `Delete chat session "${targetSession.title}"? This will remove the full message history.`,
    );
    if (!confirmed) {
      return;
    }

    setDeletingSessionId(sessionId);
    try {
      await deleteChatSession(sessionId);

      const remainingSessions = sessions.filter((session) => session.id !== sessionId);
      setSessions(remainingSessions);

      if (currentSession?.id === sessionId) {
        setCurrentSession(null);
        setMessages([]);
        setStreamingToken("");
        setStreamingMeta(null);

        if (remainingSessions[0]) {
          await selectSession(remainingSessions[0].id);
        }
      }
    } catch {
      alert("Failed to delete chat session.");
    } finally {
      setDeletingSessionId(null);
    }
  }

  async function sendMessage(event: React.FormEvent) {
    event.preventDefault();
    if (!input.trim() || !currentSession || loading) {
      return;
    }

    const currentInput = input;
    const userMessage: ChatMessage = {
      id: Math.random().toString(),
      role: "user",
      content: currentInput,
      status: "completed",
      metadata_json: null,
      created_at: new Date().toISOString(),
    };

    setMessages((current) => [...current, userMessage]);
    setInput("");
    setLoading(true);
    setStreamingToken("");
    setStreamingMeta(null);

    try {
      const token = await getAuthToken();
      const response = await fetch(
        `${API_BASE_URL}/v1/chat/sessions/${currentSession.id}/messages`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({
            content: currentInput,
            model_choice: model,
          }),
        },
      );

      if (!response.ok) {
        throw new Error("Stream failed");
      }

      const reader = response.body?.getReader();
      if (reader) {
        const decoder = new TextDecoder();
        let buffer = "";
        let accumulatedText = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            buffer += decoder.decode();
            break;
          }

          buffer += decoder.decode(value, { stream: true });
          const frames = buffer.split("\n\n");
          buffer = frames.pop() ?? "";

          for (const frame of frames) {
            const parsed = parseSseFrame(frame);
            if (!parsed) {
              continue;
            }

            if (parsed.event === "token") {
              accumulatedText += String(parsed.data);
              setStreamingToken(accumulatedText);
              continue;
            }

            if (parsed.event === "end") {
              setStreamingMeta(parsed.data);
              continue;
            }

            if (parsed.event === "error") {
              const detail = String(parsed.data);
              setStreamingToken((current) =>
                current || "The request failed before a complete response was produced.",
              );
              setStreamingMeta({
                route: "unknown",
                provider: { name: "unknown", model },
                error: detail,
                error_detail: detail,
              });
            }
          }
        }

        if (buffer.trim()) {
          const parsed = parseSseFrame(buffer);
          if (parsed?.event === "end") {
            setStreamingMeta(parsed.data);
          }
        }
      }

      await selectSession(currentSession.id);
    } catch {
      setStreamingMeta({
        route: "unknown",
        provider: { name: "unknown", model },
        error: "Failed to send message.",
        error_detail: "The browser request failed before a valid SSE response was received.",
      });
      setStreamingToken("Failed to send message.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="chatjvb-layout">
      <div className="chatjvb-sidebar">
        <div className="chatjvb-sidebar-header">
          <button className="chatjvb-new-btn" onClick={handleCreateSession}>
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M12 5v14M5 12h14" />
            </svg>
            New Chat
          </button>
        </div>
        <div
          style={{
            flex: 1,
            overflowY: "auto",
            padding: "0 12px",
            display: "flex",
            flexDirection: "column",
            gap: "4px",
          }}
        >
          {sessions.map((session) => (
            <div
              key={session.id}
              className={`chatjvb-session-item ${currentSession?.id === session.id ? "active" : ""}`}
            >
              <button
                onClick={() => void selectSession(session.id)}
                className={`chatjvb-session-btn ${currentSession?.id === session.id ? "active" : ""}`}
              >
                {session.title}
              </button>
              <button
                type="button"
                className="chatjvb-session-delete"
                disabled={deletingSessionId === session.id || (loading && currentSession?.id === session.id)}
                onClick={(event) => {
                  event.stopPropagation();
                  void handleDeleteSession(session.id);
                }}
                aria-label={`Delete ${session.title}`}
                title="Delete session"
              >
                {deletingSessionId === session.id ? "..." : "x"}
              </button>
            </div>
          ))}
        </div>
        <div className="chatjvb-bottom-nav">
          <Link href="/dashboard" className="chatjvb-bottom-link">
            Dataset Ops Dashboard
          </Link>
        </div>
      </div>

      <div className="chatjvb-main">
        <div className="chatjvb-header">
          {currentSession ? (
            <select
              className="chatjvb-model-select"
              value={model}
              onChange={(event) => setModel(normalizeModelId(event.target.value))}
            >
              {MODELS.map((entry) => (
                <option key={entry.id} value={entry.id}>
                  {entry.name}
                </option>
              ))}
            </select>
          ) : null}
        </div>

        {currentSession ? (
          <>
            <div className="chatjvb-messages">
              {messages.map((message) => (
                <div key={message.id} className="chatjvb-msg-row">
                  <div
                    className="chatjvb-msg-content"
                    style={{
                      justifyContent:
                        message.role === "user" ? "flex-end" : "flex-start",
                    }}
                  >
                    {message.role === "user" ? (
                      <div className="chatjvb-user-msg">{message.content}</div>
                    ) : (
                      <AssistantResponse
                        message={message}
                        meta={message.metadata_json}
                        content={message.content}
                      />
                    )}
                  </div>
                </div>
              ))}

              {streamingToken ? (
                <div className="chatjvb-msg-row">
                  <div className="chatjvb-msg-content">
                    <AssistantResponse
                      message={{
                        id: "streaming",
                        role: "assistant",
                        content: streamingToken,
                        status: "streaming",
                        model_name: model,
                        metadata_json: streamingMeta,
                        created_at: new Date().toISOString(),
                      }}
                      meta={streamingMeta}
                      content={streamingToken}
                    />
                  </div>
                </div>
              ) : null}

              {loading && !streamingToken ? (
                <div className="chatjvb-msg-row">
                  <div className="chatjvb-msg-content">
                    <div className="chatjvb-assistant-msg" style={{ opacity: 0.5 }}>
                      <div className="chatjvb-avatar ai">
                        <svg
                          width="18"
                          height="18"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        >
                          <circle cx="12" cy="12" r="10"></circle>
                          <line x1="12" y1="8" x2="12" y2="12"></line>
                          <line x1="12" y1="16" x2="12.01" y2="16"></line>
                        </svg>
                      </div>
                      <div style={{ flex: 1 }}>
                        <div>Processing...</div>
                        <div className="chatjvb-model-meta">
                          Model: {getDisplayModelName(model)}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ) : null}
              <div ref={messagesEndRef} />
            </div>

            <div className="chatjvb-input-wrapper">
              <form className="chatjvb-input-box" onSubmit={sendMessage}>
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="#999"
                  strokeWidth="2"
                  style={{ marginLeft: 4 }}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"></path>
                </svg>
                <input
                  type="text"
                  placeholder="Message ChatJVB..."
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                />
                <button
                  className="chatjvb-submit-btn"
                  type="submit"
                  disabled={loading || !input.trim()}
                  style={{ background: input.trim() ? "black" : "#e5e5e5" }}
                >
                  <svg
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke={input.trim() ? "white" : "#a3a3a3"}
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <line x1="12" y1="19" x2="12" y2="5"></line>
                    <polyline points="5 12 12 5 19 12"></polyline>
                  </svg>
                </button>
              </form>
            </div>
          </>
        ) : (
          <div className="chatjvb-empty">ChatJVB</div>
        )}
      </div>
    </div>
  );
}

function parseSseFrame(frame: string): { event: string; data: any } | null {
  const lines = frame.split("\n");
  let event = "message";
  let data = "";

  for (const line of lines) {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
      continue;
    }

    if (line.startsWith("data:")) {
      data += line.slice(5).trim();
    }
  }

  if (!data) {
    return null;
  }

  try {
    return { event, data: JSON.parse(data) };
  } catch {
    return { event, data };
  }
}
