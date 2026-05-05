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
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

const MODELS = [
  { id: "llama-3.3-70b-versatile", name: "Llama 3.3 70B (Cloud)" },
  { id: "gemini-2.5-flash", name: "Gemini 2.5 Flash (Cloud)" },
  { id: "gemma", name: "Gemma (Ollama Local)" },
];

type ChatMeta = Record<string, any> | null | undefined;
type RuntimeFlagTone =
  | "accent"
  | "success"
  | "warning"
  | "danger"
  | "muted";

type RuntimeFlag = {
  label: string;
  tone: RuntimeFlagTone;
};

const AGENT_ACTIVITY_LINE_RE =
  /^\s*>\s*AI\s+(?:dang\s+)?suy\s+luan:\s*(.+?)\s*$/i;

function normalizeModelId(modelId: string): string {
  if (modelId === "gemini-1.5-flash" || modelId === "gemini-1.5-flash-latest") {
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

function getResponseModelName(
  message: ChatMessage | null,
  meta: ChatMeta,
): string {
  const providerModel = meta?.provider?.model;
  if (providerModel) {
    return getDisplayModelName(providerModel);
  }
  return getDisplayModelName(message?.model_name);
}

function getFriendlyErrorMessage(error: string): string {
  const normalized = error.toLowerCase();
  if (
    normalized.includes("429") ||
    normalized.includes("resource_exhausted") ||
    normalized.includes("quota") ||
    normalized.includes("rate limit")
  ) {
    const base =
      "Hệ thống đang tạm thời quá tải hoặc bạn đã hết hạn mức (Quota) của model này.";

    // Try to extract retry time for Gemini
    const retryMatch = error.match(/retry in ([\d\.]+)s/i);
    if (retryMatch) {
      return `${base} Vui lòng thử lại sau khoảng ${Math.ceil(parseFloat(retryMatch[1]))} giây nữa nhé.`;
    }

    return `${base} Vui lòng thử lại sau ít phút hoặc thử đổi sang sử dụng model khác nhé.`;
  }
  return error;
}

function getErrorDetail(meta: ChatMeta): string | null {
  if (!meta) {
    return null;
  }
  const rawError = meta.error_detail ?? meta.error ?? null;
  return rawError ? getFriendlyErrorMessage(rawError) : null;
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

function didPevRun(meta: ChatMeta): boolean {
  if (!meta) {
    return false;
  }

  return Boolean(
    meta.route === "agent" ||
      meta.route === "agent_aborted" ||
      meta.verification ||
      meta.agent_traces?.length,
  );
}

function getRuntimeFlags(meta: ChatMeta): RuntimeFlag[] {
  const flags: RuntimeFlag[] = [];
  if (!didPevRun(meta)) {
    if (meta?.route === "sql") {
      flags.push({ label: "Direct SQL", tone: "accent" });
    } else if (meta?.route === "rag") {
      flags.push({ label: "Direct RAG", tone: "accent" });
    } else if (meta?.route === "chat") {
      flags.push({ label: "Direct Chat", tone: "muted" });
    } else if (meta?.route === "clarification") {
      flags.push({ label: "Clarification", tone: "warning" });
    }
    return flags;
  }

  const verificationStatus = meta?.verification?.status;
  const rawAttempts = Number(meta?.verification?.attempts ?? 0);
  const attempts =
    Number.isFinite(rawAttempts) && rawAttempts > 0
      ? Math.floor(rawAttempts)
      : 0;
  const stepsTaken =
    typeof meta?.steps_taken === "number" && meta.steps_taken > 0
      ? Math.floor(meta.steps_taken)
      : null;

  if (meta?.route === "agent_aborted") {
    flags.push({ label: "PEV Aborted", tone: "danger" });
  } else if (verificationStatus === "passed" && attempts > 0) {
    flags.push({ label: "PEV Self-Corrected", tone: "success" });
  } else if (verificationStatus === "passed") {
    flags.push({ label: "PEV Verified", tone: "success" });
  } else if (verificationStatus === "insufficient_evidence") {
    flags.push({ label: "PEV Safe Fallback", tone: "warning" });
  } else if (verificationStatus === "needs_revision") {
    flags.push({ label: "PEV Fallback", tone: "warning" });
  } else if (verificationStatus === "skipped") {
    flags.push({ label: "PEV Unverified", tone: "muted" });
  } else {
    flags.push({ label: "PEV Ran", tone: "accent" });
  }

  if (stepsTaken) {
    flags.push({
      label: `${stepsTaken} Step${stepsTaken === 1 ? "" : "s"}`,
      tone: "muted",
    });
  }

  if (attempts > 0) {
    flags.push({
      label: `${attempts} Revision${attempts === 1 ? "" : "s"}`,
      tone: "accent",
    });
  }

  return flags;
}

function hasProcessContent(meta: ChatMeta): boolean {
  return Boolean(
    meta &&
    (meta.route ||
      meta.provider ||
      meta.route_reason ||
      meta.route_confidence !== undefined ||
      meta.sql_used ||
      meta.error ||
      meta.error_detail ||
      meta.citations?.length ||
      meta.dataset_sources?.length ||
      meta.primary_sources?.length ||
      meta.agent_traces?.length ||
      meta.verification ||
      meta.data_preview?.length),
  );
}

function extractAgentActivity(content: string): {
  displayContent: string;
  activityLines: string[];
} {
  const activityLines: string[] = [];
  const keptLines: string[] = [];

  for (const line of content.split(/\r?\n/)) {
    const matched = line.match(AGENT_ACTIVITY_LINE_RE);
    if (matched) {
      activityLines.push(matched[1].trim());
      continue;
    }
    keptLines.push(line);
  }

  return {
    displayContent: keptLines.join("\n").replace(/\n{3,}/g, "\n\n").trim(),
    activityLines,
  };
}

function getDocumentSources(meta: ChatMeta): any[] {
  if (!Array.isArray(meta?.citations)) {
    return [];
  }
  return meta.citations;
}

function getDatasetSources(meta: ChatMeta): any[] {
  if (!Array.isArray(meta?.dataset_sources)) {
    return [];
  }
  return meta.dataset_sources;
}

function getPrimarySources(meta: ChatMeta): any[] {
  if (Array.isArray(meta?.primary_sources) && meta.primary_sources.length > 0) {
    return meta.primary_sources;
  }
  const datasetSources = getDatasetSources(meta);
  if (datasetSources.length > 0) {
    return datasetSources;
  }
  return getDocumentSources(meta);
}

function renderSourceBadges(sources: any[]) {
  if (!sources.length) {
    return null;
  }

  return (
    <div className="chatjvb-citations">
      {sources.map((source: any, index: number) => {
        const isDataset = source.kind === "dataset";
        const label = isDataset ? "DATASET " : "DOC ";
        const filename =
          source.original_filename ?? source.title ?? source.schema_name ?? "Unknown";
        const suffix =
          !isDataset && source.source_page ? ` (p.${source.source_page})` : "";
        const keyBase =
          source.asset_id ??
          source.chunk_id ??
          source.original_filename ??
          source.title ??
          "source";

        return (
          <div
            key={`${keyBase}-${index}`}
            className="chatjvb-badge"
            title={source.quote}
          >
            {label}
            {filename}
            {suffix}
          </div>
        );
      })}
    </div>
  );
}

function ProcessSnapshot({
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
  const primarySources = getPrimarySources(meta);

  return (
    <>
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

      {primarySources.length > 0 ? (
        <div className="chatjvb-process-section">
          <div className="chatjvb-process-label">Primary Sources</div>
          {renderSourceBadges(primarySources)}
        </div>
      ) : null}

      {Array.isArray(meta?.agent_traces) && meta.agent_traces.length > 0 ? (
        <div className="chatjvb-process-section">
          <div className="chatjvb-process-label">Agent Steps</div>
          <div className="chatjvb-process-traces">
            {meta.agent_traces.map((trace: any, index: number) => (
              <div
                key={`${trace.tool ?? "tool"}-${index}`}
                className="chatjvb-process-trace"
              >
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
    </>
  );
}

function ActivityPanel({
  message,
  meta,
  activityLines,
}: {
  message: ChatMessage | null;
  meta: ChatMeta;
  activityLines: string[];
}) {
  const hasProcess = hasProcessContent(meta);
  if (!activityLines.length && !hasProcess) {
    return null;
  }

  const summaryLabel = activityLines.length > 0 ? "AI suy luan" : "Process";
  const summaryMeta =
    activityLines.length > 0
      ? `${activityLines.length} update${activityLines.length === 1 ? "" : "s"}`
      : "Details";
  const shouldOpen =
    message?.status === "streaming" &&
    activityLines.length > 0 &&
    !meta?.verification;

  return (
    <details className="chatjvb-process" open={shouldOpen}>
      <summary className="chatjvb-process-summary">
        <span>{summaryLabel}</span>
        <span className="chatjvb-process-summary-meta">{summaryMeta}</span>
      </summary>
      <div className="chatjvb-process-body">
        {activityLines.length > 0 ? (
          <div className="chatjvb-process-section">
            <div className="chatjvb-process-label">Tool Activity</div>
            <div className="chatjvb-agent-activity-list">
              {activityLines.map((line, index) => (
                <div
                  key={`${line}-${index}`}
                  className="chatjvb-agent-activity-item"
                >
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {line}
                  </ReactMarkdown>
                </div>
              ))}
            </div>
          </div>
        ) : null}

        {hasProcess ? (
          <div className="chatjvb-process-section">
            <div className="chatjvb-process-label">Process</div>
            <ProcessSnapshot message={message} meta={meta} />
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
  const runtimeFlags = getRuntimeFlags(meta);
  const { displayContent, activityLines } = extractAgentActivity(content);

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
      <div style={{ flex: 1, minWidth: 0 }}>
        {displayContent ? (
          <div className="chatjvb-markdown">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {displayContent}
            </ReactMarkdown>
          </div>
        ) : null}
        <div className="chatjvb-model-meta">Model: {modelName}</div>
        {runtimeFlags.length > 0 ? (
          <div className="chatjvb-runtime-flags">
            {runtimeFlags.map((flag, index) => (
              <div
                key={`${flag.label}-${index}`}
                className={`chatjvb-runtime-flag chatjvb-runtime-flag--${flag.tone}`}
              >
                {flag.label}
              </div>
            ))}
          </div>
        ) : null}
        {errorDetail ? (
          <div className="chatjvb-inline-error">{errorDetail}</div>
        ) : null}
        <ActivityPanel
          message={message}
          meta={meta}
          activityLines={activityLines}
        />
      </div>
    </div>
  );
}

export default function ChatView() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSession, setCurrentSession] = useState<ChatSession | null>(
    null,
  );
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [model, setModel] = useState(() => normalizeModelId(MODELS[0].id));
  const [loading, setLoading] = useState(false);
  const [deletingSessionId, setDeletingSessionId] = useState<string | null>(
    null,
  );
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

      const sortedMessages = (session.messages || []).sort((a, b) => {
        const diff =
          new Date(a.created_at).getTime() - new Date(b.created_at).getTime();
        if (diff !== 0) return diff;
        if (a.role === "user" && b.role !== "user") return -1;
        if (a.role !== "user" && b.role === "user") return 1;
        return 0;
      });

      setMessages(sortedMessages);
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

      const remainingSessions = sessions.filter(
        (session) => session.id !== sessionId,
      );
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
              setStreamingToken(
                (current) =>
                  current ||
                  "The request failed before a complete response was produced.",
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
        error_detail:
          "The browser request failed before a valid SSE response was received.",
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
                disabled={
                  deletingSessionId === session.id ||
                  (loading && currentSession?.id === session.id)
                }
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
              onChange={(event) =>
                setModel(normalizeModelId(event.target.value))
              }
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
                    <div
                      className="chatjvb-assistant-msg"
                      style={{ opacity: 0.5 }}
                    >
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
