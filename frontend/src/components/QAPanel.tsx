import { useEffect, useState } from "react";
import { ShieldCheck, ShieldAlert, Info } from "lucide-react";
import { api } from "../api/client";

interface QAData {
  summary_text: string;
  missingness: Record<string, { n_missing: number; pct_missing: number }>;
  llm_rules: Array<{ field: string; rule: string; severity: string; n_fail: number; passed: boolean | null }>;
}

export default function QAPanel() {
  const [qa, setQA] = useState<QAData | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api.qa().then(setQA).catch((e) => setError(e.message));
  }, []);

  if (error) return <div style={{ color: "var(--danger)", padding: 12, fontSize: 12 }}>{error}</div>;
  if (!qa) return <div style={{ padding: 12, color: "var(--muted)", fontSize: 12 }}>Loading QA report…</div>;

  const totalMissing = Object.values(qa.missingness).reduce((s, v) => s + v.n_missing, 0);
  const passedRules = qa.llm_rules.filter((r) => r.passed === true);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      {/* Summary */}
      <div style={{ background: "var(--surface2)", borderRadius: 8, padding: "10px 12px" }}>
        <div style={{ fontSize: 10, color: "var(--muted)", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.06em" }}>Summary</div>
        <pre style={{ fontFamily: "ui-monospace, monospace", fontSize: 11, color: "var(--text)", whiteSpace: "pre-wrap" }}>
          {qa.summary_text}
        </pre>
      </div>

      {/* Missingness */}
      <div>
        <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 6, display: "flex", alignItems: "center", gap: 6 }}>
          <Info size={11} />
          Missingness by column
          {totalMissing === 0 && <span className="badge badge-green">All clean</span>}
        </div>
        {Object.entries(qa.missingness).map(([col, v]) => (
          <div key={col} style={{ display: "flex", justifyContent: "space-between", padding: "3px 0", fontSize: 11 }}>
            <span style={{ color: "var(--muted)" }}>{col}</span>
            <span style={{ color: v.n_missing > 0 ? "var(--warn)" : "var(--accent2)", fontVariantNumeric: "tabular-nums" }}>
              {v.n_missing} ({v.pct_missing}%)
            </span>
          </div>
        ))}
      </div>

      {/* LLM-proposed rules */}
      {qa.llm_rules.length > 0 && (
        <div>
          <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 6, display: "flex", alignItems: "center", gap: 6 }}>
            <ShieldCheck size={11} color="var(--accent2)" />
            AI-proposed QA rules ({passedRules.length}/{qa.llm_rules.length} passed)
          </div>
          {qa.llm_rules.map((r, i) => (
            <div key={i} style={{
              display: "flex", gap: 8, padding: "5px 0",
              borderBottom: "1px solid var(--border)", alignItems: "flex-start",
            }}>
              {r.passed === true ? (
                <ShieldCheck size={12} color="var(--accent2)" style={{ flexShrink: 0, marginTop: 1 }} />
              ) : (
                <ShieldAlert size={12} color="var(--warn)" style={{ flexShrink: 0, marginTop: 1 }} />
              )}
              <div style={{ flex: 1 }}>
                <span style={{ fontSize: 11, color: r.passed ? "var(--text)" : "var(--warn)" }}>
                  {r.field}: {r.rule}
                </span>
                {r.n_fail > 0 && (
                  <span style={{ fontSize: 10, color: "var(--muted)", marginLeft: 6 }}>{r.n_fail} violations</span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
      {qa.llm_rules.length === 0 && (
        <div style={{ fontSize: 11, color: "var(--muted)", fontStyle: "italic" }}>
          No LLM rules — add GROQ_API_KEY to enable AI-proposed data validation.
        </div>
      )}
    </div>
  );
}
