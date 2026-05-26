import { useEffect, useState } from "react";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";
import { api } from "../api/client";

export default function SignalBadge() {
  const [signal, setSignal] = useState<{ action: string; signal_z: number; date: string } | null>(null);

  useEffect(() => {
    api.signal().then((d) => {
      if (d.signals.length > 0) setSignal(d.signals[d.signals.length - 1]);
    }).catch(() => {});
  }, []);

  if (!signal) return null;

  const colorMap: Record<string, string> = { LONG: "var(--accent2)", SHORT: "var(--danger)", FLAT: "var(--muted)" };
  const IconMap: Record<string, typeof TrendingUp> = { LONG: TrendingUp, SHORT: TrendingDown, FLAT: Minus };
  const Icon = IconMap[signal.action] || Minus;
  const color = colorMap[signal.action] || "var(--muted)";

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, background: "var(--surface2)", borderRadius: 8, padding: "8px 12px" }}>
      <Icon size={14} color={color} />
      <div>
        <div style={{ fontSize: 10, color: "var(--muted)" }}>Trading Signal ({signal.date})</div>
        <div style={{ fontSize: 14, fontWeight: 700, color }}>{signal.action}</div>
      </div>
      <div style={{ marginLeft: "auto", textAlign: "right" }}>
        <div style={{ fontSize: 10, color: "var(--muted)" }}>Z-score</div>
        <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text)", fontVariantNumeric: "tabular-nums" }}>
          {signal.signal_z?.toFixed(2)}
        </div>
      </div>
    </div>
  );
}
