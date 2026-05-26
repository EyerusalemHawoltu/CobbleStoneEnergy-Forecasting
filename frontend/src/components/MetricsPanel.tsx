import { useEffect, useState } from "react";
import { TrendingUp } from "lucide-react";
import { api } from "../api/client";

interface Metrics {
  cv: Record<string, number>;
  test: Record<string, { name: string; mae: number; rmse: number; tail_mae_p90: number }>;
  test_period: { start: string; end: string; n_hours: number };
  demo_mode: boolean;
}

export default function MetricsPanel() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api.metrics().then(setMetrics).catch((e) => setError(e.message));
  }, []);

  if (error) return <div style={{ color: "var(--danger)", padding: 12, fontSize: 12 }}>{error}</div>;
  if (!metrics) return <div style={{ padding: 12, color: "var(--muted)", fontSize: 12 }}>Loading metrics…</div>;

  const models = Object.values(metrics.test);
  const lgbm = metrics.test["lgbm"];
  const naive = metrics.test["naive"];
  const improvement = naive && lgbm ? ((naive.mae - lgbm.mae) / naive.mae * 100).toFixed(1) : null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      {/* Demo mode banner */}
      {metrics.demo_mode && (
        <div style={{ background: "rgba(245,158,11,0.1)", border: "1px solid rgba(245,158,11,0.3)", borderRadius: 8, padding: "8px 12px", fontSize: 11, color: "var(--warn)" }}>
          Demo mode — using synthetic data. Add ENTSOE_API_KEY for real market data.
        </div>
      )}

      {/* Test period */}
      <div style={{ fontSize: 11, color: "var(--muted)" }}>
        Test period: {metrics.test_period.start} → {metrics.test_period.end} ({metrics.test_period.n_hours.toLocaleString()} hours)
      </div>

      {/* CV summary */}
      <div style={{ background: "var(--surface2)", borderRadius: 8, padding: "10px 12px", display: "flex", gap: 20 }}>
        <div>
          <div style={{ fontSize: 10, color: "var(--muted)", marginBottom: 2 }}>CV MAE</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: "var(--accent)" }}>{metrics.cv.mae}</div>
          <div style={{ fontSize: 10, color: "var(--muted)" }}>EUR/MWh</div>
        </div>
        <div>
          <div style={{ fontSize: 10, color: "var(--muted)", marginBottom: 2 }}>CV RMSE</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: "var(--accent)" }}>{metrics.cv.rmse}</div>
          <div style={{ fontSize: 10, color: "var(--muted)" }}>EUR/MWh</div>
        </div>
        <div>
          <div style={{ fontSize: 10, color: "var(--muted)", marginBottom: 2 }}>Tail P90</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: "var(--warn)" }}>{metrics.cv.tail_mae_p90}</div>
          <div style={{ fontSize: 10, color: "var(--muted)" }}>EUR/MWh</div>
        </div>
        {improvement && (
          <div style={{ marginLeft: "auto", textAlign: "right" }}>
            <div style={{ fontSize: 10, color: "var(--muted)", marginBottom: 2 }}>vs Naive</div>
            <div style={{ fontSize: 16, fontWeight: 700, color: "var(--accent2)" }}>−{improvement}%</div>
            <div style={{ fontSize: 10, color: "var(--muted)" }}>MAE</div>
          </div>
        )}
      </div>

      {/* Model comparison table */}
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
        <thead>
          <tr style={{ color: "var(--muted)", fontSize: 10, textTransform: "uppercase" }}>
            <th style={{ textAlign: "left", padding: "4px 0", fontWeight: 500 }}>Model</th>
            <th style={{ textAlign: "right", padding: "4px 6px", fontWeight: 500 }}>MAE</th>
            <th style={{ textAlign: "right", padding: "4px 6px", fontWeight: 500 }}>RMSE</th>
            <th style={{ textAlign: "right", padding: "4px 0", fontWeight: 500 }}>Tail</th>
          </tr>
        </thead>
        <tbody>
          {models.map((m, i) => (
            <tr key={i} style={{ borderTop: "1px solid var(--border)" }}>
              <td style={{ padding: "6px 0", color: m.name === "LightGBM" ? "var(--accent)" : "var(--text)" }}>
                {i === 0 && <TrendingUp size={11} style={{ marginRight: 4, display: "inline", color: "var(--accent)" }} />}
                {m.name}
              </td>
              <td style={{ textAlign: "right", padding: "6px 6px", fontVariantNumeric: "tabular-nums" }}>{m.mae}</td>
              <td style={{ textAlign: "right", padding: "6px 6px", fontVariantNumeric: "tabular-nums" }}>{m.rmse}</td>
              <td style={{ textAlign: "right", padding: "6px 0", color: "var(--warn)", fontVariantNumeric: "tabular-nums" }}>{m.tail_mae_p90}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div style={{ fontSize: 10, color: "var(--muted)" }}>MAE/RMSE/Tail in EUR/MWh · Tail = mean absolute error on top-10% price hours</div>
    </div>
  );
}
