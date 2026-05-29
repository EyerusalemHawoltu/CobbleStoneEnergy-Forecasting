import { useState, useEffect } from "react";
import { Activity, BarChart2, MessageCircle, ShieldCheck, TrendingUp } from "lucide-react";
import ChatPanel from "./components/ChatPanel";
import ForecastChart from "./components/ForecastChart";
import MetricsPanel from "./components/MetricsPanel";
import QAPanel from "./components/QAPanel";
import SignalBadge from "./components/SignalBadge";
import ForwardDeliveryPanel from "./components/ForwardDeliveryPanel";
import { api } from "./api/client";
import "./index.css";

type Tab = "metrics" | "delivery" | "qa";

interface ChartState {
  type: "hourly" | "delivery";
  data: unknown[];
  title: string;
}

export default function App() {
  const [ready, setReady] = useState<boolean | null>(null);
  const [demoMode, setDemoMode] = useState(false);
  const [activeTab, setActiveTab] = useState<Tab>("metrics");
  const [mobilePanel, setMobilePanel] = useState<"chat" | Tab>("chat");
  const [chart, setChart] = useState<ChartState | null>(null);

  useEffect(() => {
    const poll = setInterval(() => {
      api.health().then((h) => {
        if (h.pipeline_ready) {
          setReady(true);
          setDemoMode(h.demo_mode);
          clearInterval(poll);
        } else {
          setReady(false);
        }
      }).catch(() => setReady(false));
    }, 2000);
    return () => clearInterval(poll);
  }, []);

  const handleData = (data: Record<string, unknown>) => {
    const type = data.type as string;
    if (type === "get_daily_forecast" && data.hourly) {
      setChart({ type: "hourly", data: data.hourly as unknown[], title: `Hourly Forecast — ${data.date}` });
    } else if (type === "get_delivery_summary" && data.data) {
      setChart({ type: "delivery", data: data.data as unknown[], title: `${(data.period_type as string)} Delivery Curve` });
    } else if (type === "get_forward_delivery") {
      const rows = [];
      if (data.next_week) rows.push({ period: "Next Week", ...(data.next_week as Record<string, unknown>) });
      if (data.next_month) rows.push({ period: "Next Month", ...(data.next_month as Record<string, unknown>) });
      if (rows.length) setChart({ type: "delivery", data: rows, title: "Forward Delivery Curve — Next Week & Next Month" });
    }
  };

  return (
    <div style={{ height: "100vh", display: "flex", flexDirection: "column", overflow: "hidden" }}>
      {/* Header */}
      <header style={{
        display: "flex", alignItems: "center", gap: 12, padding: "0 20px",
        height: 52, background: "var(--surface)", borderBottom: "1px solid var(--border)",
        flexShrink: 0,
      }}>
        <Activity size={18} color="var(--accent)" />
        <span className="header-title" style={{ fontWeight: 700, fontSize: 15, letterSpacing: "-0.02em" }}>
          Cobblestone Energy — DE Power Forecast
        </span>
        <span className="badge badge-blue" style={{ marginLeft: 4 }}>Groq Free</span>
        {demoMode && <span className="badge badge-yellow">Demo Mode</span>}
        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
          {ready === null && <span style={{ fontSize: 11, color: "var(--muted)" }}>Connecting…</span>}
          {ready === false && <span style={{ fontSize: 11, color: "var(--warn)" }}>Pipeline loading…</span>}
          {ready === true && <span style={{ fontSize: 11, color: "var(--accent2)" }}>● Live</span>}
        </div>
      </header>

      {/* Main layout */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>
        {/* Left sidebar — data panels */}
        <aside className={`app-sidebar${mobilePanel !== "chat" ? " mobile-visible" : ""}`} style={{
          width: 320, flexShrink: 0, background: "var(--surface)",
          borderRight: "1px solid var(--border)",
          flexDirection: "column", overflow: "hidden",
        }}>
          {/* Signal */}
          <div style={{ padding: "12px 16px", borderBottom: "1px solid var(--border)" }}>
            {ready && <SignalBadge />}
          </div>

          {/* Tabs */}
          <div style={{ display: "flex", borderBottom: "1px solid var(--border)" }}>
            {([["metrics", BarChart2, "Performance"], ["delivery", TrendingUp, "Forward Curve"], ["qa", ShieldCheck, "Data QA"]] as const).map(([tab, Icon, label]) => (
              <button key={tab} onClick={() => { setActiveTab(tab); setMobilePanel(tab); }} style={{
                flex: 1, background: "none", border: "none", padding: "10px 0",
                cursor: "pointer", fontSize: 12, fontWeight: 500,
                color: activeTab === tab ? "var(--accent)" : "var(--muted)",
                borderBottom: activeTab === tab ? "2px solid var(--accent)" : "2px solid transparent",
                display: "flex", alignItems: "center", justifyContent: "center", gap: 6,
                transition: "color 0.15s",
              }}>
                <Icon size={13} />
                {label}
              </button>
            ))}
          </div>

          {/* Tab content */}
          <div style={{ flex: 1, overflowY: "auto", padding: "14px 16px" }}>
            {ready === true ? (
              activeTab === "metrics" ? <MetricsPanel /> : activeTab === "delivery" ? <ForwardDeliveryPanel /> : <QAPanel />
            ) : (
              <div style={{ color: "var(--muted)", fontSize: 12, padding: "20px 0", textAlign: "center" }}>
                {ready === false ? "Pipeline initialising… (this takes ~7 min on first run)" : "Connecting to backend…"}
              </div>
            )}
          </div>
        </aside>

        {/* Centre — chat + charts */}
        <main className={`app-main${mobilePanel !== "chat" ? " panel-open" : ""}`} style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
          {/* Chart area */}
          {chart && (
            <div style={{ padding: "12px 16px", borderBottom: "1px solid var(--border)", flexShrink: 0 }}>
              <ForecastChart type={chart.type} data={chart.data as any} title={chart.title} />
            </div>
          )}

          {/* Chat panel */}
          <div style={{ flex: 1, overflow: "hidden" }}>
            <ChatPanel onDataReceived={handleData} />
          </div>
        </main>
      </div>

      {/* Bottom nav — visible on mobile only (CSS hides it on desktop) */}
      <nav className="bottom-nav">
        {([
          ["chat",     MessageCircle, "Chat"],
          ["metrics",  BarChart2,     "Performance"],
          ["delivery", TrendingUp,    "Curve"],
          ["qa",       ShieldCheck,   "QA"],
        ] as const).map(([panel, Icon, label]) => (
          <button
            key={panel}
            className={`bottom-nav-btn${mobilePanel === panel ? " active" : ""}`}
            onClick={() => { setMobilePanel(panel); if (panel !== "chat") setActiveTab(panel); }}
          >
            <Icon size={20} />
            <span>{label}</span>
          </button>
        ))}
      </nav>
    </div>
  );
}
