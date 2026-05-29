import { useEffect, useState } from "react";
import { api } from "../api/client";

interface PeriodView {
  label: string;
  period: string;
  base_avg: number;
  peak_avg: number;
  offpeak_avg: number;
  n_days: number;
  daily: Array<{ date: string; base_avg: number; peak_avg: number; offpeak_avg: number }>;
}

interface ForwardData {
  next_week: PeriodView;
  next_month: PeriodView;
}

function fmt(v: number | null | undefined) {
  if (v == null) return "—";
  return v.toFixed(1);
}

function PeriodCard({ data }: { data: PeriodView }) {
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ fontSize: 11, fontWeight: 700, color: "var(--accent)", marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.06em" }}>
        {data.label}
      </div>
      <div style={{ fontSize: 10, color: "var(--muted)", marginBottom: 6 }}>{data.period}</div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 4 }}>
        {[
          { label: "Base", value: data.base_avg, color: "var(--fg)" },
          { label: "Peak", value: data.peak_avg, color: "#f59e0b" },
          { label: "Off-pk", value: data.offpeak_avg, color: "var(--muted)" },
        ].map(({ label, value, color }) => (
          <div key={label} style={{ background: "var(--bg)", borderRadius: 6, padding: "6px 8px", textAlign: "center" }}>
            <div style={{ fontSize: 10, color: "var(--muted)", marginBottom: 2 }}>{label}</div>
            <div style={{ fontSize: 13, fontWeight: 700, color }}>{fmt(value)}</div>
            <div style={{ fontSize: 9, color: "var(--muted)" }}>€/MWh</div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function ForwardDeliveryPanel() {
  const [data, setData] = useState<ForwardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.forwardDelivery()
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div style={{ fontSize: 11, color: "var(--muted)", padding: "8px 0" }}>Loading forward curve…</div>;
  if (error) return <div style={{ fontSize: 11, color: "var(--warn)", padding: "8px 0" }}>Error: {error}</div>;
  if (!data) return null;

  return (
    <div>
      <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 10, lineHeight: 1.5 }}>
        Forward averages derived from the DA forecast model — base, peak (08–19 CET), off-peak.
      </div>
      <PeriodCard data={data.next_week} />
      <PeriodCard data={data.next_month} />
    </div>
  );
}
