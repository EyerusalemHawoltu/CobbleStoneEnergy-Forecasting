import { useState, useRef, useEffect } from "react";
import { Send, Bot, User, Loader2, Zap } from "lucide-react";
import { api } from "../api/client";

interface Message {
  id: number;
  role: "user" | "assistant";
  content: string;
  toolCalls?: Array<{ tool: string; args: Record<string, unknown> }>;
  data?: Record<string, unknown> | null;
  latency?: number;
}

const SUGGESTIONS = [
  "What's the forecast for tomorrow?",
  "Show me the forward curve",
  "Show me the monthly delivery curve",
  "How accurate is the LightGBM model?",
  "What's the current trading signal?",
  "Run data quality checks",
];

interface Props {
  onDataReceived: (data: Record<string, unknown>) => void;
}

export default function ChatPanel({ onDataReceived }: Props) {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 0,
      role: "assistant",
      content:
        "Hello! I'm your German power market analyst, powered by Groq's Llama 3.3 70B (free). Ask me about day-ahead price forecasts, model performance, data quality, or trading signals.",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const history = messages
    .slice(-6)
    .map((m) => ({ role: m.role, content: m.content }));

  const send = async (text: string) => {
    if (!text.trim() || loading) return;
    const userMsg: Message = { id: Date.now(), role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const result = await api.chat(text, history);
      const assistantMsg: Message = {
        id: Date.now() + 1,
        role: "assistant",
        content: result.response,
        toolCalls: result.tool_calls?.map((tc) => ({ tool: tc.tool, args: tc.args })),
        data: result.data,
        latency: result.latency_s,
      };
      setMessages((prev) => [...prev, assistantMsg]);
      if (result.data) onDataReceived(result.data);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          role: "assistant",
          content: `Error: ${err instanceof Error ? err.message : "Request failed"}`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
      {/* Messages */}
      <div style={{ flex: 1, overflowY: "auto", padding: "16px", display: "flex", flexDirection: "column", gap: "12px" }}>
        {messages.map((msg) => (
          <div key={msg.id} style={{ display: "flex", gap: "10px", alignItems: "flex-start", flexDirection: msg.role === "user" ? "row-reverse" : "row" }}>
            <div style={{
              width: 30, height: 30, borderRadius: "50%", flexShrink: 0,
              background: msg.role === "user" ? "var(--accent)" : "var(--surface2)",
              display: "flex", alignItems: "center", justifyContent: "center",
            }}>
              {msg.role === "user" ? <User size={14} /> : <Bot size={14} color="var(--accent2)" />}
            </div>
            <div style={{ maxWidth: "80%", display: "flex", flexDirection: "column", gap: 4 }}>
              <div style={{
                background: msg.role === "user" ? "var(--accent)" : "var(--surface2)",
                padding: "10px 14px", borderRadius: msg.role === "user" ? "12px 12px 4px 12px" : "12px 12px 12px 4px",
                fontSize: 13, lineHeight: 1.6, whiteSpace: "pre-wrap",
              }}>
                {msg.content}
              </div>
              {msg.toolCalls && msg.toolCalls.length > 0 && (
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                  {msg.toolCalls.map((tc, i) => (
                    <span key={i} className="badge badge-blue" style={{ fontSize: 10 }}>
                      <Zap size={9} style={{ marginRight: 3, display: "inline" }} />
                      {tc.tool}
                    </span>
                  ))}
                  {msg.latency && <span className="badge badge-green" style={{ fontSize: 10 }}>{msg.latency}s</span>}
                </div>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
            <div style={{ width: 30, height: 30, borderRadius: "50%", background: "var(--surface2)", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <Bot size={14} color="var(--accent2)" />
            </div>
            <div style={{ background: "var(--surface2)", padding: "10px 14px", borderRadius: "12px 12px 12px 4px", display: "flex", gap: 6, alignItems: "center" }}>
              <Loader2 size={14} style={{ animation: "spin 1s linear infinite" }} />
              <span style={{ color: "var(--muted)", fontSize: 12 }}>Thinking…</span>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Suggestions — always visible */}
      {!loading && (
        <div style={{ padding: "0 16px 12px", display: "flex", flexWrap: "wrap", gap: 6 }}>
          {SUGGESTIONS.map((s) => (
            <button key={s} onClick={() => send(s)} style={{
              background: "var(--surface2)", border: "1px solid var(--border)",
              color: "var(--muted)", borderRadius: 20, padding: "5px 12px",
              fontSize: 12, cursor: "pointer", transition: "all 0.15s",
            }}
              onMouseEnter={(e) => { (e.target as HTMLElement).style.color = "var(--text)"; (e.target as HTMLElement).style.borderColor = "var(--accent)"; }}
              onMouseLeave={(e) => { (e.target as HTMLElement).style.color = "var(--muted)"; (e.target as HTMLElement).style.borderColor = "var(--border)"; }}
            >
              {s}
            </button>
          ))}
        </div>
      )}

      {/* Input */}
      <div style={{ padding: "12px 16px", borderTop: "1px solid var(--border)", display: "flex", gap: 8 }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && send(input)}
          placeholder="Ask about forecasts, signals, model accuracy…"
          disabled={loading}
          style={{
            flex: 1, background: "var(--surface2)", border: "1px solid var(--border)",
            borderRadius: 8, padding: "9px 14px", color: "var(--text)", fontSize: 13,
            outline: "none", transition: "border-color 0.15s",
          }}
          onFocus={(e) => (e.target.style.borderColor = "var(--accent)")}
          onBlur={(e) => (e.target.style.borderColor = "var(--border)")}
        />
        <button onClick={() => send(input)} disabled={loading || !input.trim()} style={{
          background: "var(--accent)", border: "none", borderRadius: 8,
          width: 38, height: 38, cursor: "pointer", display: "flex",
          alignItems: "center", justifyContent: "center", flexShrink: 0,
          opacity: loading || !input.trim() ? 0.5 : 1, transition: "opacity 0.15s",
        }}>
          <Send size={15} color="#fff" />
        </button>
      </div>

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
