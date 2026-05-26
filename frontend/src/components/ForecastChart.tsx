import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, BarChart, Bar, Legend,
} from "recharts";

interface HourlyPoint { hour: number; forecast: number; actual: number | null; is_peak: boolean }
interface DeliveryPoint { period: string; base_avg: number | null; peak_avg: number | null; offpeak_avg: number | null; p10?: number | null; p90?: number | null }

interface Props {
  type: "hourly" | "delivery";
  data: HourlyPoint[] | DeliveryPoint[];
  title?: string;
}

const fmt = (v: number) => `${v.toFixed(1)} €`;

export default function ForecastChart({ type, data, title }: Props) {
  if (!data || data.length === 0) return null;

  return (
    <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10, padding: "16px" }}>
      {title && <div style={{ fontSize: 12, fontWeight: 600, color: "var(--muted)", marginBottom: 12, textTransform: "uppercase", letterSpacing: "0.06em" }}>{title}</div>}

      {type === "hourly" && (
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart data={data as HourlyPoint[]} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="fg" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#4f8ef7" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#4f8ef7" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis dataKey="hour" tick={{ fontSize: 10, fill: "var(--muted)" }} tickFormatter={(h) => `${h}h`} />
            <YAxis tick={{ fontSize: 10, fill: "var(--muted)" }} tickFormatter={fmt} width={54} />
            <Tooltip
              contentStyle={{ background: "var(--surface2)", border: "1px solid var(--border)", borderRadius: 6, fontSize: 12 }}
              formatter={(v, name) => [`${(v as number).toFixed(2)} EUR/MWh`, name === "forecast" ? "Forecast" : "Actual"]}
              labelFormatter={(h) => `Hour ${h}:00 CET`}
            />
            <ReferenceLine y={0} stroke="var(--border)" />
            <Area type="monotone" dataKey="forecast" stroke="#4f8ef7" fill="url(#fg)" strokeWidth={2} dot={false} name="forecast" />
            <Area type="monotone" dataKey="actual" stroke="#34d399" fill="none" strokeWidth={1.5} strokeDasharray="4 2" dot={false} name="actual" />
          </AreaChart>
        </ResponsiveContainer>
      )}

      {type === "delivery" && (
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={data as DeliveryPoint[]} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis dataKey="period" tick={{ fontSize: 9, fill: "var(--muted)" }} tickFormatter={(v: string) => v.slice(0, 7)} />
            <YAxis tick={{ fontSize: 10, fill: "var(--muted)" }} tickFormatter={fmt} width={54} />
            <Tooltip
              contentStyle={{ background: "var(--surface2)", border: "1px solid var(--border)", borderRadius: 6, fontSize: 12 }}
              formatter={(v) => [`${(v as number)?.toFixed(2)} EUR/MWh`]}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Bar dataKey="base_avg" fill="#4f8ef7" name="Base" radius={[2, 2, 0, 0]} />
            <Bar dataKey="peak_avg" fill="#34d399" name="Peak" radius={[2, 2, 0, 0]} />
            <Bar dataKey="offpeak_avg" fill="#8892b0" name="Off-peak" radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
