"use client";

import { useState, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  Legend,
  ReferenceLine,
  BarChart,
  Bar,
} from "recharts";

type Metrics = {
  total_return: number;
  annualized_return?: number;
  sharpe_ratio: number;
  sortino_ratio?: number;
  volatility: number;
  max_drawdown: number;
  n_rebalances?: number;
};

type Benchmarks = {
  helix: Metrics;
  helix_asym?: Metrics | null;
  ew7: Metrics | null;
  spy: Metrics | null;
};

type PeriodData = {
  period: string;
  start_date: string;
  end_date: string;
  metrics: Metrics;
  benchmarks?: Benchmarks;
  portfolio_values: [string, number][];
  asym_values?: [string, number][];
  ew7_values?: [string, number][];
  spy_values?: [string, number][];
  weights_history: { date: string; weights: Record<string, number> }[];
  rebalance_dates: string[];
};

type BacktestData = {
  periods: PeriodData[];
};

type SJMMetricsData = {
  start_date: string;
  end_date: string;
  factors: string[];
  daily: { date: string; [k: string]: string | number | null | undefined }[];
  sharpe_per_factor: Record<string, number>;
  cum_pnl_per_factor: Record<string, number[]>;
  cum_pnl_dates: string[];
};

const FACTOR_COLORS: Record<string, string> = {
  QUAL: "#79c0ff",
  MTUM: "#d2a8ff",
  USMV: "#7ee787",
  VLUE: "#ffa657",
  SIZE: "#ff7b72",
  IWF: "#a5d6ff",
};

function formatPct(val: number) {
  return (val * 100).toFixed(2) + "%";
}

function generateSyntheticCurve(
  startDate: string,
  endDate: string,
  totalReturn: number
): [string, number][] {
  const start = new Date(startDate).getTime();
  const end = new Date(endDate).getTime();
  const days = Math.ceil((end - start) / (24 * 60 * 60 * 1000));
  const points = Math.min(252, Math.max(60, Math.floor(days / 2)));
  const growth = 1 + totalReturn;
  const dailyGrowth = Math.pow(growth, 1 / points);
  const result: [string, number][] = [];
  let value = 100;
  for (let i = 0; i <= points; i++) {
    const t = i / points;
    const date = new Date(start + t * (end - start));
    result.push([date.toISOString().slice(0, 10), value]);
    value *= dailyGrowth * (1 + Math.sin(i * 0.3) * 0.003);
  }
  return result;
}

/** Merge portfolio series. Helix (Sym) is primary timeline; others aligned by date (ffill). */
function mergePortfolioSeries(
  helix: [string, number][],
  asym?: [string, number][],
  ew7?: [string, number][],
  spy?: [string, number][],
  showAsym?: boolean,
  showEw7?: boolean,
  showSpy?: boolean
): { date: string; "Helix (Sym)": number; "Helix (Asym)"?: number; EW7?: number; SPY?: number }[] {
  const asymMap = new Map<string, number>();
  const ew7Map  = new Map<string, number>();
  const spyMap  = new Map<string, number>();
  const baseA = asym?.[0]?.[1] ?? 1;
  const baseE = ew7?.[0]?.[1]  ?? 1;
  const baseS = spy?.[0]?.[1]  ?? 1;
  asym?.forEach(([d, v]) => asymMap.set(d.slice(0, 10), (v / baseA) * 100));
  ew7?.forEach(([d, v])  => ew7Map.set(d.slice(0, 10),  (v / baseE) * 100));
  spy?.forEach(([d, v])  => spyMap.set(d.slice(0, 10),  (v / baseS) * 100));

  let lastA: number | undefined;
  let lastE: number | undefined;
  let lastS: number | undefined;
  return helix.map(([dateStr, val]) => {
    const date = dateStr.slice(0, 10);
    const base = helix[0]?.[1] ?? 1;
    const out: { date: string; "Helix (Sym)": number; "Helix (Asym)"?: number; EW7?: number; SPY?: number } = {
      date,
      "Helix (Sym)": (val / base) * 100,
    };
    if (showAsym && asym?.length) {
      const v = asymMap.get(date);
      if (v != null) lastA = v;
      if (lastA != null) out["Helix (Asym)"] = lastA;
    }
    if (showEw7 && ew7?.length) {
      const v = ew7Map.get(date);
      if (v != null) lastE = v;
      if (lastE != null) out.EW7 = lastE;
    }
    if (showSpy && spy?.length) {
      const v = spyMap.get(date);
      if (v != null) lastS = v;
      if (lastS != null) out.SPY = lastS;
    }
    return out;
  });
}

function PortfolioChart({
  data,
  asymData,
  ew7Data,
  spyData,
  showAsym,
  showEw7,
  showSpy,
  onToggleAsym,
  onToggleEw7,
  onToggleSpy,
  fallback,
  startDate,
  endDate,
}: {
  data: [string, number][];
  asymData?: [string, number][];
  ew7Data?: [string, number][];
  spyData?: [string, number][];
  showAsym: boolean;
  showEw7: boolean;
  showSpy: boolean;
  onToggleAsym: (v: boolean) => void;
  onToggleEw7: (v: boolean) => void;
  onToggleSpy: (v: boolean) => void;
  fallback?: { start_date: string; end_date: string; total_return: number };
  startDate?: string;
  endDate?: string;
}) {
  const chartData = data.length
    ? data
    : fallback
      ? generateSyntheticCurve(
          fallback.start_date,
          fallback.end_date,
          fallback.total_return
        )
      : [];

  if (!chartData.length) {
    return (
      <div className="empty-state">
        No portfolio data. Run{" "}
        <code>python analyze_strategy.py --export --quick</code> to generate.
      </div>
    );
  }

  const filtered =
    startDate && endDate
      ? chartData.filter(([d]) => d >= startDate && d <= endDate)
      : chartData;
  if (!filtered.length) {
    return <div className="empty-state">No data in period {startDate} to {endDate}</div>;
  }
  const base = filtered[0]?.[1] ?? 1;
  const helixNorm = filtered.map(([date, val]) => [date, (val / base) * 100] as [string, number]);
  const plotData = mergePortfolioSeries(
    helixNorm,
    asymData,
    ew7Data,
    spyData,
    showAsym && !!asymData?.length,
    showEw7 && !!ew7Data?.length,
    showSpy && !!spyData?.length
  );

  const hasOverlays = (asymData?.length ?? 0) > 0 || (ew7Data?.length ?? 0) > 0 || (spyData?.length ?? 0) > 0;

  return (
    <div className="portfolio-chart-wrap">
      {hasOverlays && (
        <div className="chart-toggles">
          <label className="toggle-label">
            <input
              type="checkbox"
              checked={showAsym}
              onChange={(e) => onToggleAsym(e.target.checked)}
              disabled={!asymData?.length}
            />
            <span>Helix (Asym)</span>
          </label>
          <label className="toggle-label">
            <input
              type="checkbox"
              checked={showEw7}
              onChange={(e) => onToggleEw7(e.target.checked)}
              disabled={!ew7Data?.length}
            />
            <span>EW(7)</span>
          </label>
          <label className="toggle-label">
            <input
              type="checkbox"
              checked={showSpy}
              onChange={(e) => onToggleSpy(e.target.checked)}
              disabled={!spyData?.length}
            />
            <span>SPY</span>
          </label>
        </div>
      )}
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={plotData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="date"
            tickFormatter={(v) => {
              const d = new Date(v);
              return d.toLocaleDateString("en-US", { month: "short", year: "2-digit" });
            }}
          />
          <YAxis
            domain={["auto", "auto"]}
            tickFormatter={(v) => v + "%"}
            width={45}
          />
          <Tooltip
            contentStyle={{
              background: "var(--bg-elevated)",
              border: "1px solid var(--border)",
              borderRadius: "6px",
            }}
            labelFormatter={(label) => new Date(label).toLocaleDateString()}
            formatter={(value: number, name: string) => [value != null ? value.toFixed(1) + "%" : "-", name]}
          />
          <ReferenceLine y={100} stroke="var(--text-muted)" strokeDasharray="2 2" strokeOpacity={0.5} />
          <Line
            type="monotone"
            dataKey="Helix (Sym)"
            stroke="var(--accent)"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
          {showAsym && asymData?.length && (
            <Line
              type="monotone"
              dataKey="Helix (Asym)"
              stroke="#f78166"
              strokeWidth={2}
              strokeDasharray="5 2"
              dot={false}
              isAnimationActive={false}
            />
          )}
          {showEw7 && ew7Data?.length && (
            <Line
              type="monotone"
              dataKey="EW7"
              stroke="var(--chart-minvol)"
              strokeWidth={1.5}
              strokeDasharray="4 2"
              dot={false}
              isAnimationActive={false}
            />
          )}
          {showSpy && spyData?.length && (
            <Line
              type="monotone"
              dataKey="SPY"
              stroke="var(--chart-value)"
              strokeWidth={1.5}
              strokeDasharray="2 2"
              dot={false}
              isAnimationActive={false}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

/** Build time series of weights over OOS period. Weights carry forward until next rebalance. */
function buildWeightsTimeSeries(
  startDate: string,
  endDate: string,
  weightsHistory: PeriodData["weights_history"]
): { date: string; [k: string]: string | number }[] {
  if (!weightsHistory.length) return [];

  const factors = Object.keys(weightsHistory[0].weights).sort();
  const sorted = [...weightsHistory].sort(
    (a, b) => new Date(a.date).getTime() - new Date(b.date).getTime()
  );

  const points: { date: string; [k: string]: string | number }[] = [];
  const start = new Date(startDate);
  const end = new Date(endDate);

  // Point at period start: equal weight until first rebalance (or first rb weights if rb is on start)
  const firstRb = sorted[0];
  const isFirstRbOnStart = firstRb.date <= startDate;
  const startWeights = isFirstRbOnStart
    ? firstRb.weights
    : Object.fromEntries(factors.map((f) => [f, 1 / factors.length]));
  points.push({
    date: startDate,
    ...Object.fromEntries(
      factors.map((f) => [f, Math.round((startWeights[f] ?? 1 / 6) * 1000) / 1000])
    ),
  });

  // Each rebalance date
  for (const { date, weights } of sorted) {
    if (date >= startDate && date <= endDate) {
      points.push({
        date,
        ...Object.fromEntries(
          factors.map((f) => [f, Math.round((weights[f] ?? 0) * 1000) / 1000])
        ),
      });
    }
  }

  // Point at period end: carry forward last rebalance
  if (points.length > 1) {
    const lastWeights = sorted[sorted.length - 1].weights;
    points.push({
      date: endDate,
      ...Object.fromEntries(
        factors.map((f) => [f, Math.round((lastWeights[f] ?? 0) * 1000) / 1000])
      ),
    });
  }

  return points;
}

function AllocationChart({
  startDate,
  endDate,
  weightsHistory,
}: {
  startDate: string;
  endDate: string;
  weightsHistory: PeriodData["weights_history"];
}) {
  const chartData = buildWeightsTimeSeries(startDate, endDate, weightsHistory);

  if (!chartData.length) {
    return (
      <div className="empty-state">
        No allocation data for this period.
      </div>
    );
  }

  const factors = Object.keys(chartData[0]).filter((k) => k !== "date").sort();

  return (
    <ResponsiveContainer width="100%" height={280}>
      <AreaChart
        data={chartData}
        margin={{ top: 10, right: 20, left: 0, bottom: 0 }}
        stackOffset="expand"
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis
          dataKey="date"
          tickFormatter={(v) => {
            const d = new Date(v);
            return d.toLocaleDateString("en-US", { month: "short", year: "2-digit" });
          }}
        />
        <YAxis domain={[0, 1]} tickFormatter={(v) => (v * 100).toFixed(0) + "%"} width={45} />
        <Tooltip
          contentStyle={{
            background: "var(--bg-elevated)",
            border: "1px solid var(--border)",
            borderRadius: "6px",
          }}
          formatter={(value: number, name: string) => [(value * 100).toFixed(1) + "%", name]}
          labelFormatter={(label) => new Date(label).toLocaleDateString()}
        />
        <Legend />
        {factors.map((f) => (
          <Area
            key={f}
            type="monotone"
            dataKey={f}
            stackId="1"
            stroke={FACTOR_COLORS[f] ?? "#8b949e"}
            fill={FACTOR_COLORS[f] ?? "#8b949e"}
            fillOpacity={0.85}
            isAnimationActive={false}
          />
        ))}
      </AreaChart>
    </ResponsiveContainer>
  );
}

function SJMMetricsSection({ data }: { data: SJMMetricsData | null }) {
  if (!data) {
    return (
      <section className="section">
        <h2>SJM metrics (paper-aligned)</h2>
        <div className="chart-card">
          <div className="empty-state">
            To see per-factor long-short Sharpe, cumulative PnL, and regime over time, run from repo root:
            <pre className="export-code">
              python scripts/export_sjm_metrics_series.py --start 2017-01-01 --end 2025-12-31 -c
              hyperparam/sjm_hyperparameters_best.json -o dashboard/public/sjm_metrics_series.json
            </pre>
            Then reload the dashboard.
          </div>
        </div>
      </section>
    );
  }
  const { factors, sharpe_per_factor, cum_pnl_per_factor, cum_pnl_dates, daily } = data;
  const barData = factors.map((f) => ({ factor: f, sharpe: sharpe_per_factor[f] ?? 0 }));
  const cumPnlChartData = cum_pnl_dates.map((date, i) => {
    const row: { date: string; [k: string]: number | string } = { date };
    factors.forEach((f) => {
      row[f] = cum_pnl_per_factor[f]?.[i] ?? null;
    });
    return row;
  });
  const regimeChartData = daily.map((row) => {
    const out: { date: string; [k: string]: number | string } = { date: row.date };
    factors.forEach((f) => {
      const r = row[`${f}_regime`];
      out[f] = typeof r === "number" ? r : Number(r) || 0;
    });
    return out;
  });

  return (
    <section className="section">
      <h2>SJM metrics (paper-aligned)</h2>
      <p className="chart-caption">
        Per-factor hypothetical long-short strategy (position from regime expected active return, ±5% cap; T+2).
        Data: {data.start_date} → {data.end_date}. Generate with{" "}
        <code>python scripts/export_sjm_metrics_series.py -c hyperparam/sjm_hyperparameters_best.json -o dashboard/public/sjm_metrics_series.json</code>
      </p>
      <div className="chart-card">
        <h3>Long-short Sharpe per factor</h3>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={barData} margin={{ top: 10, right: 20, left: 10, bottom: 24 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="factor" />
            <YAxis />
            <Tooltip formatter={(v: number) => [v.toFixed(3), "Sharpe"]} />
            <Bar dataKey="sharpe" fill="var(--accent)" name="Sharpe" />
            <ReferenceLine y={0} stroke="var(--text-muted)" />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="chart-card">
        <h3>Cumulative PnL (base 100)</h3>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={cumPnlChartData} margin={{ top: 10, right: 20, left: 50, bottom: 24 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" tickFormatter={(v) => v.slice(0, 7)} />
            <YAxis />
            <Tooltip labelFormatter={(v) => v} />
            <Legend />
            {factors.map((f) => (
              <Line
                key={f}
                type="monotone"
                dataKey={f}
                stroke={FACTOR_COLORS[f] ?? "#8b949e"}
                dot={false}
                isAnimationActive={false}
                connectNulls
              />
            ))}
            <ReferenceLine y={100} stroke="var(--text-muted)" strokeDasharray="2 2" />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="chart-card">
        <h3>Regime over time (0 / 1)</h3>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={regimeChartData} margin={{ top: 10, right: 20, left: 10, bottom: 24 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" tickFormatter={(v) => v.slice(0, 7)} />
            <YAxis domain={[0, 1]} tickFormatter={(v) => (v === 1 ? "1" : "0")} />
            <Tooltip />
            <Legend />
            {factors.map((f) => (
              <Line
                key={f}
                type="monotone"
                dataKey={f}
                stroke={FACTOR_COLORS[f] ?? "#8b949e"}
                dot={false}
                isAnimationActive={false}
                connectNulls
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

function ComparisonTable({ periods }: { periods: PeriodData[] }) {
  const hasBenchmarks = periods.some((p) => p.benchmarks?.ew7 || p.benchmarks?.spy);
  const hasAsym = periods.some((p) => p.benchmarks?.helix_asym);

  if (!hasBenchmarks) {
    return (
      <div className="empty-state">
        Run <code>python analyze_strategy.py --export</code> to include EW7 and SPY benchmarks.
      </div>
    );
  }

  const cols = hasAsym ? 4 : 3;

  return (
    <div className="comparison-table-wrap">
      <table className="comparison-table">
        <thead>
          <tr>
            <th>Period</th>
            <th colSpan={cols} className="group-start">Total Return</th>
            <th colSpan={cols} className="group-start">Sharpe</th>
            <th colSpan={cols} className="group-start">Sortino</th>
            <th colSpan={cols} className="group-start">Max DD</th>
          </tr>
          <tr>
            <th></th>
            <th className="group-start">Helix (Sym)</th>
            {hasAsym && <th>Helix (Asym)</th>}
            <th>EW7</th>
            <th>SPY</th>
            <th className="group-start">Helix (Sym)</th>
            {hasAsym && <th>Helix (Asym)</th>}
            <th>EW7</th>
            <th>SPY</th>
            <th className="group-start">Helix (Sym)</th>
            {hasAsym && <th>Helix (Asym)</th>}
            <th>EW7</th>
            <th>SPY</th>
            <th className="group-start">Helix (Sym)</th>
            {hasAsym && <th>Helix (Asym)</th>}
            <th>EW7</th>
            <th>SPY</th>
          </tr>
        </thead>
        <tbody>
          {periods.map((p) => {
            const h = p.benchmarks?.helix ?? p.metrics;
            const a = p.benchmarks?.helix_asym;
            const e = p.benchmarks?.ew7;
            const s = p.benchmarks?.spy;
            const retCls = (m: Metrics | null | undefined) =>
              m ? (m.total_return >= 0 ? "positive" : "negative") : "na";
            const fmt = (m: Metrics | null | undefined, fn: (x: Metrics) => string) =>
              m ? fn(m) : "—";
            return (
              <tr key={p.period}>
                <td className="period-cell">{p.period}</td>
                <td className={`group-start ${retCls(h)}`}>{formatPct(h.total_return)}</td>
                {hasAsym && <td className={retCls(a)}>{fmt(a, (x) => formatPct(x.total_return))}</td>}
                <td className={retCls(e)}>{fmt(e, (x) => formatPct(x.total_return))}</td>
                <td className={retCls(s)}>{fmt(s, (x) => formatPct(x.total_return))}</td>
                <td className="group-start">{h.sharpe_ratio.toFixed(2)}</td>
                {hasAsym && <td className={a ? "" : "na"}>{fmt(a, (x) => x.sharpe_ratio.toFixed(2))}</td>}
                <td className={e ? "" : "na"}>{fmt(e, (x) => x.sharpe_ratio.toFixed(2))}</td>
                <td className={s ? "" : "na"}>{fmt(s, (x) => x.sharpe_ratio.toFixed(2))}</td>
                <td className="group-start">{h.sortino_ratio != null ? h.sortino_ratio.toFixed(2) : "—"}</td>
                {hasAsym && <td className={a ? "" : "na"}>{fmt(a, (x) => x.sortino_ratio != null ? x.sortino_ratio.toFixed(2) : "—")}</td>}
                <td className={e ? "" : "na"}>{fmt(e, (x) => x.sortino_ratio != null ? x.sortino_ratio.toFixed(2) : "—")}</td>
                <td className={s ? "" : "na"}>{fmt(s, (x) => x.sortino_ratio != null ? x.sortino_ratio.toFixed(2) : "—")}</td>
                <td className={`group-start negative`}>{formatPct(h.max_drawdown)}</td>
                {hasAsym && <td className={a ? "negative" : "na"}>{fmt(a, (x) => formatPct(x.max_drawdown))}</td>}
                <td className={e ? "negative" : "na"}>{fmt(e, (x) => formatPct(x.max_drawdown))}</td>
                <td className={s ? "negative" : "na"}>{fmt(s, (x) => formatPct(x.max_drawdown))}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function MetricsGrid({ m }: { m: Metrics }) {
  return (
    <div className="metrics-grid">
      <div className="metric">
        <span className="metric-label">Total Return</span>
        <span className={`metric-value ${m.total_return >= 0 ? "positive" : "negative"}`}>
          {formatPct(m.total_return)}
        </span>
      </div>
      <div className="metric">
        <span className="metric-label">Ann. Return</span>
        <span className={`metric-value ${(m.annualized_return ?? m.total_return) >= 0 ? "positive" : "negative"}`}>
          {formatPct(m.annualized_return ?? m.total_return)}
        </span>
      </div>
      <div className="metric">
        <span className="metric-label">Sharpe Ratio</span>
        <span className="metric-value">{m.sharpe_ratio.toFixed(2)}</span>
      </div>
      <div className="metric">
        <span className="metric-label">Sortino Ratio</span>
        <span className="metric-value">
          {m.sortino_ratio != null ? m.sortino_ratio.toFixed(2) : "—"}
        </span>
      </div>
      <div className="metric">
        <span className="metric-label">Volatility</span>
        <span className="metric-value">{formatPct(m.volatility)}</span>
      </div>
      <div className="metric">
        <span className="metric-label">Max Drawdown</span>
        <span className="metric-value negative">{formatPct(m.max_drawdown)}</span>
      </div>
      <div className="metric">
        <span className="metric-label">Rebalances</span>
        <span className="metric-value">{m.n_rebalances ?? "—"}</span>
      </div>
    </div>
  );
}

type TrainStreamEvent =
  | { type: "log"; line: string }
  | { type: "loss"; factor: string; iter: number; value: number }
  | { type: "done"; line: string }
  | { type: "error"; line: string }
  | { type: "exit"; code: number };

type HyperparamConfig = {
  results?: Record<string, { lambda?: number; kappa_sq?: number }>;
  metadata?: {
    validation_start?: string;
    validation_end?: string;
    holdout_start?: string;
    holdout_end?: string;
  };
};

function TrainTab() {
  const [configFiles, setConfigFiles] = useState<string[]>([]);
  const [selectedConfig, setSelectedConfig] = useState("");
  const [configContent, setConfigContent] = useState<HyperparamConfig | null>(null);
  const [jumpPenalty, setJumpPenalty] = useState(50);
  const [sparsityParam, setSparsityParam] = useState(9.5);
  const [trainStart, setTrainStart] = useState("2015-01-01");
  const [trainEnd, setTrainEnd] = useState("2023-12-31");
  const [oosStart, setOosStart] = useState("2024-01-01");
  const [oosEnd, setOosEnd] = useState("2024-12-31");
  const [logLines, setLogLines] = useState<string[]>([]);
  const [lossByFactor, setLossByFactor] = useState<Record<string, { iteration: number; value: number }[]>>({});
  const [running, setRunning] = useState(false);

  useEffect(() => {
    fetch("/api/hyperparam-configs")
      .then((r) => r.json())
      .then((d: { files?: string[] }) => setConfigFiles(d.files ?? []))
      .catch(() => setConfigFiles([]));
  }, []);

  useEffect(() => {
    if (!selectedConfig) {
      setConfigContent(null);
      return;
    }
    fetch("/api/hyperparam-configs?config=" + encodeURIComponent(selectedConfig))
      .then((r) => r.json())
      .then((data: HyperparamConfig) => {
        setConfigContent(data);
        const m = data.metadata;
        if (m?.validation_start) setTrainStart(m.validation_start);
        if (m?.validation_end) setTrainEnd(m.validation_end);
        if (m?.holdout_start) setOosStart(m.holdout_start);
        if (m?.holdout_end) setOosEnd(m.holdout_end);
      })
      .catch(() => setConfigContent(null));
  }, [selectedConfig]);

  const lockParams = !!selectedConfig && !!configContent?.results;
  const lockDates = !!selectedConfig && !!configContent?.metadata?.validation_start;

  const runTraining = () => {
    setLogLines([]);
    setLossByFactor({});
    setRunning(true);
    fetch("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        jump_penalty: jumpPenalty,
        sparsity_param: sparsityParam,
        train_start: trainStart,
        train_end: trainEnd,
        oos_start: oosStart || undefined,
        oos_end: oosEnd || undefined,
        config_path: selectedConfig ? "hyperparam/" + selectedConfig : undefined,
      }),
    })
      .then((res) => {
        if (!res.ok || !res.body) throw new Error("Stream failed");
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buf = "";
        function read() {
          reader.read().then(({ done, value }) => {
            if (done) {
              setRunning(false);
              return;
            }
            buf += decoder.decode(value, { stream: true });
            const lines = buf.split("\n");
            buf = lines.pop() || "";
            for (const line of lines) {
              if (line.startsWith("data: ")) {
                try {
                  const ev: TrainStreamEvent = JSON.parse(line.slice(6));
                  if (ev.type === "log") {
                    setLogLines((prev) => [...prev, ev.line]);
                  } else if (ev.type === "loss") {
                    setLossByFactor((prev) => ({
                      ...prev,
                      [ev.factor]: [...(prev[ev.factor] ?? []), { iteration: ev.iter, value: ev.value }],
                    }));
                  } else if (ev.type === "done" || ev.type === "error") {
                    setLogLines((prev) => [...prev, ev.line]);
                  }
                } catch (_) {}
              }
            }
            read();
          });
        }
        read();
      })
      .catch((e) => {
        setLogLines((prev) => [...prev, "Error: " + (e.message || String(e))]);
        setRunning(false);
      });
  };

  const factors = ["QUAL", "MTUM", "USMV", "VLUE", "SIZE", "IWF"];
  // One row per iteration; each factor's line has (iteration, objective). X = iteration, Y = SJM objective.
  const maxIter = Math.max(
    0,
    ...Object.values(lossByFactor).map((arr) => (arr.length ? arr[arr.length - 1].iteration : 0))
  );
  const lossChartData =
    maxIter >= 0
      ? Array.from({ length: maxIter + 1 }, (_, iteration) => {
          const row: { iteration: number; [k: string]: number | undefined } = { iteration };
          factors.forEach((f) => {
            const point = lossByFactor[f]?.find((p) => p.iteration === iteration);
            if (point != null) row[f] = point.value;
          });
          return row;
        })
      : [];

  return (
    <div className="train-tab">
      <section className="section">
        <h2>Train your own SJM and reallocate ETFs</h2>
        <p className="train-desc">
          Set hyperparameters and date ranges, then run training. Logs and loss curve stream in real time.
          Runs are logged to MLflow (experiment <code>helix-sjm-train</code>); from repo root run <code>mlflow ui</code> to view.
          If you see &quot;No module named &apos;numpy&apos;&quot;, set <code>HELIX_PYTHON</code> in <code>dashboard/.env</code> to your conda env’s Python path (see <code>dashboard/.env.example</code>).
        </p>
        <div className="train-form">
          <div className="form-row form-row-full">
            <label>Hyperparam config (optional)</label>
            <select
              value={selectedConfig}
              onChange={(e) => setSelectedConfig(e.target.value)}
              disabled={running}
            >
              <option value="">None (use λ, κ² below)</option>
              {configFiles.map((f) => (
                <option key={f} value={f}>
                  {f}
                </option>
              ))}
            </select>
            {lockParams && (
              <span className="form-hint">λ and κ² locked from config (per-factor).</span>
            )}
            {lockDates && (
              <span className="form-hint">Dates locked from config metadata.</span>
            )}
          </div>
          <div className="form-row">
            <label>λ (jump penalty)</label>
            <input
              type="number"
              min={1}
              max={200}
              step={1}
              value={jumpPenalty}
              onChange={(e) => setJumpPenalty(parseFloat(e.target.value) || 50)}
              disabled={running || lockParams}
              title={lockParams ? "Locked by selected config" : undefined}
            />
          </div>
          <div className="form-row">
            <label>κ² (sparsity)</label>
            <input
              type="number"
              min={1}
              max={30}
              step={0.5}
              value={sparsityParam}
              onChange={(e) => setSparsityParam(parseFloat(e.target.value) || 9.5)}
              disabled={running || lockParams}
              title={lockParams ? "Locked by selected config" : undefined}
            />
          </div>
          <div className="form-row">
            <label>Training start</label>
            <input
              type="date"
              value={trainStart}
              onChange={(e) => setTrainStart(e.target.value)}
              disabled={running || lockDates}
              title={lockDates ? "Locked by selected config" : undefined}
            />
          </div>
          <div className="form-row">
            <label>Training end</label>
            <input
              type="date"
              value={trainEnd}
              onChange={(e) => setTrainEnd(e.target.value)}
              disabled={running || lockDates}
              title={lockDates ? "Locked by selected config" : undefined}
            />
          </div>
          <div className="form-row">
            <label>OOS start (optional)</label>
            <input
              type="date"
              value={oosStart}
              onChange={(e) => setOosStart(e.target.value)}
              disabled={running || lockDates}
              title={lockDates ? "Locked by selected config" : undefined}
            />
          </div>
          <div className="form-row">
            <label>OOS end (optional)</label>
            <input
              type="date"
              value={oosEnd}
              onChange={(e) => setOosEnd(e.target.value)}
              disabled={running || lockDates}
              title={lockDates ? "Locked by selected config" : undefined}
            />
          </div>
          <button
            type="button"
            className="train-run-btn"
            onClick={runTraining}
            disabled={running}
          >
            {running ? "Running…" : "Run training"}
          </button>
        </div>
      </section>
      <section className="section">
        <h3>Live loss curve</h3>
        <p className="chart-caption">
          <strong>X = iteration</strong> (training step for each factor). <strong>Y = SJM objective</strong> (loss; lower is better). One line per factor; factors train one after another so each line grows as its run progresses.
        </p>
        <div className="chart-card">
          {lossChartData.length ? (
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={lossChartData} margin={{ top: 10, right: 20, left: 50, bottom: 24 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="iteration"
                  label={{ value: "Iteration", position: "insideBottom", offset: -4 }}
                  allowDecimals={false}
                />
                <YAxis
                  label={{ value: "SJM objective", angle: -90, position: "insideLeft", offset: 0 }}
                  domain={["auto", "auto"]}
                />
                <Tooltip
                  formatter={(value: number) => [value?.toFixed(4) ?? "—", ""]}
                  labelFormatter={(label) => "Iteration " + label}
                />
                <Legend />
                {factors.map((f) => (
                  <Line
                    key={f}
                    type="monotone"
                    dataKey={f}
                    stroke={FACTOR_COLORS[f] ?? "#8b949e"}
                    dot={false}
                    isAnimationActive={false}
                    connectNulls
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="empty-state">Start a run to see the loss curve here.</div>
          )}
        </div>
      </section>
      <section className="section">
        <h3>Terminal output</h3>
        <div className="terminal-wrap">
          <pre className="terminal">
            {logLines.length ? logLines.join("\n") : "Submit the form to stream training and allocation logs."}
          </pre>
        </div>
      </section>
    </div>
  );
}

export default function Dashboard() {
  const [tab, setTab] = useState<"backtest" | "train">("backtest");
  const [data, setData] = useState<BacktestData | null>(null);
  const [sjmMetrics, setSjmMetrics] = useState<SJMMetricsData | null>(null);
  const [selectedPeriod, setSelectedPeriod] = useState<string>("");
  const [showAsym, setShowAsym] = useState(true);
  const [showEw7, setShowEw7] = useState(false);
  const [showSpy, setShowSpy] = useState(false);
  const [loading, setLoading] = useState(true);

  const showTrainTab = process.env.NEXT_PUBLIC_ENABLE_TRAIN_TAB === "true";

  useEffect(() => {
    if (!showTrainTab && tab === "train") setTab("backtest");
  }, [showTrainTab, tab]);

  useEffect(() => {
    fetch("/" + (process.env.NEXT_PUBLIC_BACKTEST_JSON || "backtest_data.json"))
      .then((r) => r.json())
      .then((d: BacktestData) => {
        setData(d);
        if (d.periods?.length && !selectedPeriod) {
          setSelectedPeriod(d.periods[0].period);
        }
      })
      .catch((e) => {
        console.error("Failed to load backtest data:", e);
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    fetch("/sjm_metrics_series.json")
      .then((r) => (r.ok ? r.json() : null))
      .then((d: SJMMetricsData | null) => setSjmMetrics(d))
      .catch(() => setSjmMetrics(null));
  }, []);

  const periodData = data?.periods?.find((p) => p.period === selectedPeriod);
  const hasBacktestData = !!data?.periods?.length;

  if (loading) {
    return (
      <main className="main">
        <div className="loading">Loading...</div>
      </main>
    );
  }

  return (
    <main className="main">
      <header className="header">
        <div>
          <h1>Helix Factor Strategy</h1>
          <p className="subtitle">Active rebalancing across factor regimes</p>
        </div>
        <div className="header-right">
          <nav className="tabs">
            <button
              type="button"
              className={tab === "backtest" ? "tab active" : "tab"}
              onClick={() => setTab("backtest")}
            >
              Backtest
            </button>
            {showTrainTab && (
              <button
                type="button"
                className={tab === "train" ? "tab active" : "tab"}
                onClick={() => setTab("train")}
              >
                Train your own SJM
              </button>
            )}
          </nav>
          {tab === "backtest" && hasBacktestData && (
            <div className="period-select">
              <label htmlFor="period">Period</label>
              <select
                id="period"
                value={selectedPeriod}
                onChange={(e) => setSelectedPeriod(e.target.value)}
              >
                {data.periods.map((p) => (
                  <option key={p.period} value={p.period}>
                    {p.period} ({p.start_date} → {p.end_date})
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>
      </header>

      {showTrainTab && tab === "train" && <TrainTab />}

      {tab === "backtest" && !hasBacktestData && (
        <div className="empty-state full">
          No backtest data. Run <code>python analyze_strategy.py --export --quick</code> from the
          project root.
        </div>
      )}

      {tab === "backtest" && periodData && (
        <>
          <section className="section">
            <h2>Helix (Sym) vs Helix (Asym) vs EW(7) vs SPY — All Periods</h2>
            <div className="chart-card">
              <ComparisonTable periods={data!.periods} />
            </div>
          </section>

          <section className="section metrics-section">
            <h2>Performance ({selectedPeriod})</h2>
            <MetricsGrid
              m={(() => {
                const m = periodData.metrics;
                if (
                  periodData.portfolio_values.length > 0 &&
                  periodData.start_date &&
                  periodData.end_date
                ) {
                  const filtered = periodData.portfolio_values.filter(
                    ([d]) => d >= periodData.start_date && d <= periodData.end_date
                  );
                  if (filtered.length >= 2) {
                    const first = filtered[0][1];
                    const last = filtered[filtered.length - 1][1];
                    const periodReturn = (last - first) / first;
                    const years = filtered.length / 252;
                    const periodAnnRet =
                      years > 0 ? Math.pow(1 + periodReturn, 1 / years) - 1 : periodReturn;
                    return {
                      ...m,
                      total_return: periodReturn,
                      annualized_return: periodAnnRet,
                    };
                  }
                }
                return m;
              })()}
            />
          </section>

          <section className="section">
            <h2>Portfolio Value (normalized to 100)</h2>
            <div className="chart-card">
              <PortfolioChart
                data={periodData.portfolio_values}
                asymData={periodData.asym_values}
                ew7Data={periodData.ew7_values}
                spyData={periodData.spy_values}
                showAsym={showAsym}
                showEw7={showEw7}
                showSpy={showSpy}
                onToggleAsym={setShowAsym}
                onToggleEw7={setShowEw7}
                onToggleSpy={setShowSpy}
                startDate={periodData.start_date}
                endDate={periodData.end_date}
                fallback={
                  periodData.portfolio_values.length === 0
                    ? {
                        start_date: periodData.start_date,
                        end_date: periodData.end_date,
                        total_return: periodData.metrics.total_return,
                      }
                    : undefined
                }
              />
            </div>
          </section>

          <section className="section">
            <h2>ETF Allocation Over OOS Period (stacked, normalized to 1)</h2>
            <div className="chart-card">
              <AllocationChart
                startDate={periodData.start_date}
                endDate={periodData.end_date}
                weightsHistory={periodData.weights_history}
              />
            </div>
          </section>

          <SJMMetricsSection data={sjmMetrics} />
        </>
      )}

      <footer className="footer">
        <span>Dynamic Factor Allocation • Regime-Switching</span>
      </footer>
    </main>
  );
}
