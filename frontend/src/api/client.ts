const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function request<T>(path: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

export const api = {
  health: () => request<{ status: string; pipeline_ready: boolean; demo_mode: boolean }>("/api/health"),

  chat: (message: string, history?: Array<{ role: string; content: string }>) =>
    request<{
      response: string;
      tool_calls: Array<{ tool: string; args: Record<string, unknown>; result: unknown }>;
      data: Record<string, unknown> | null;
      latency_s?: number;
    }>("/api/chat", {
      method: "POST",
      body: JSON.stringify({ message, history }),
    }),

  dailyForecast: (date: string) =>
    request<{
      date: string;
      summary: { base_avg: number; peak_avg: number; offpeak_avg: number; p10: number; p90: number };
      hourly: Array<{ hour: number; utc_timestamp: string; forecast: number; actual: number | null; is_peak: boolean }>;
    }>(`/api/forecast/daily/${date}`),

  delivery: (period: "daily" | "weekly" | "monthly") =>
    request<{ period_type: string; data: Array<Record<string, number | null | string>> }>(`/api/forecast/delivery?period=${period}`),

  metrics: () =>
    request<{
      cv: Record<string, number>;
      test: Record<string, { name: string; mae: number; rmse: number; tail_mae_p90: number }>;
      test_period: { start: string; end: string; n_hours: number };
      demo_mode: boolean;
    }>("/api/forecast/metrics"),

  signal: () =>
    request<{ signals: Array<{ date: string; forecast_price: number; benchmark_price: number; signal_z: number; action: string }> }>("/api/forecast/signal"),

  qa: () =>
    request<{
      summary_text: string;
      missingness: Record<string, { n_missing: number; pct_missing: number }>;
      llm_rules: Array<{ field: string; rule: string; severity: string; n_fail: number; passed: boolean }>;
    }>("/api/qa/report"),
};
