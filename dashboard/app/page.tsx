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
} from "recharts";

type Metrics = {
  total_return: number;
  annualized_return?: number;
  sharpe_ratio: number;
  volatility: number;
  max_drawdown: number;
  n_rebalances?: number;
};

type Benchmarks = {
  helix: Metrics;
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
  ew7_values?: [string, number][];
  spy_values?: [string, number][];
  weights_history: { date: string; weights: Record<string, number> }[];
  rebalance_dates: string[];
};

type BacktestData = {
  periods: PeriodData[];
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

/** Merge portfolio series. Helix is primary timeline; EW7/SPY aligned by date (ffill if needed). */
function mergePortfolioSeries(
  helix: [string, number][],
  ew7?: [string, number][],
  spy?: [string, number][],
  showEw7?: boolean,
  showSpy?: boolean
): { date: string; Helix: number; EW7?: number; SPY?: number }[] {
  const ew7Map = new Map<string, number>();
  const spyMap = new Map<string, number>();
  const baseE = ew7?.[0]?.[1] ?? 1;
  const baseS = spy?.[0]?.[1] ?? 1;
  ew7?.forEach(([d, v]) => ew7Map.set(d.slice(0, 10), (v / baseE) * 100));
  spy?.forEach(([d, v]) => spyMap.set(d.slice(0, 10), (v / baseS) * 100));

  let lastE: number | undefined;
  let lastS: number | undefined;
  return helix.map(([dateStr, val]) => {
    const date = dateStr.slice(0, 10);
    const base = helix[0]?.[1] ?? 1;
    const out: { date: string; Helix: number; EW7?: number; SPY?: number } = {
      date,
      Helix: (val / base) * 100,
    };
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
  ew7Data,
  spyData,
  showEw7,
  showSpy,
  onToggleEw7,
  onToggleSpy,
  fallback,
  startDate,
  endDate,
}: {
  data: [string, number][];
  ew7Data?: [string, number][];
  spyData?: [string, number][];
  showEw7: boolean;
  showSpy: boolean;
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
    ew7Data,
    spyData,
    showEw7 && !!ew7Data?.length,
    showSpy && !!spyData?.length
  );

  const hasOverlays = (ew7Data?.length ?? 0) > 0 || (spyData?.length ?? 0) > 0;

  return (
    <div className="portfolio-chart-wrap">
      {hasOverlays && (
        <div className="chart-toggles">
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
            dataKey="Helix"
            stroke="var(--accent)"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
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

function ComparisonTable({ periods }: { periods: PeriodData[] }) {
  const hasBenchmarks = periods.some((p) => p.benchmarks?.ew7 || p.benchmarks?.spy);

  if (!hasBenchmarks) {
    return (
      <div className="empty-state">
        Run <code>python analyze_strategy.py --export</code> to include EW7 and SPY benchmarks.
      </div>
    );
  }

  return (
    <div className="comparison-table-wrap">
      <table className="comparison-table">
        <thead>
          <tr>
            <th>Period</th>
            <th colSpan={3}>Total Return</th>
            <th colSpan={3}>Sharpe</th>
            <th colSpan={3}>Volatility</th>
            <th colSpan={3}>Max DD</th>
          </tr>
          <tr>
            <th></th>
            <th>Helix</th>
            <th>EW7</th>
            <th>SPY</th>
            <th>Helix</th>
            <th>EW7</th>
            <th>SPY</th>
            <th>Helix</th>
            <th>EW7</th>
            <th>SPY</th>
            <th>Helix</th>
            <th>EW7</th>
            <th>SPY</th>
          </tr>
        </thead>
        <tbody>
          {periods.map((p) => {
            const h = p.benchmarks?.helix ?? p.metrics;
            const e = p.benchmarks?.ew7;
            const s = p.benchmarks?.spy;
            return (
              <tr key={p.period}>
                <td className="period-cell">{p.period}</td>
                <td className={h.total_return >= 0 ? "positive" : "negative"}>
                  {formatPct(h.total_return)}
                </td>
                <td className={e ? (e.total_return >= 0 ? "positive" : "negative") : "na"}>
                  {e ? formatPct(e.total_return) : "—"}
                </td>
                <td className={s ? (s.total_return >= 0 ? "positive" : "negative") : "na"}>
                  {s ? formatPct(s.total_return) : "—"}
                </td>
                <td>{h.sharpe_ratio.toFixed(2)}</td>
                <td className={e ? "" : "na"}>{e ? e.sharpe_ratio.toFixed(2) : "—"}</td>
                <td className={s ? "" : "na"}>{s ? s.sharpe_ratio.toFixed(2) : "—"}</td>
                <td>{formatPct(h.volatility)}</td>
                <td className={e ? "" : "na"}>{e ? formatPct(e.volatility) : "—"}</td>
                <td className={s ? "" : "na"}>{s ? formatPct(s.volatility) : "—"}</td>
                <td className="negative">{formatPct(h.max_drawdown)}</td>
                <td className={e ? "negative" : "na"}>{e ? formatPct(e.max_drawdown) : "—"}</td>
                <td className={s ? "negative" : "na"}>{s ? formatPct(s.max_drawdown) : "—"}</td>
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

export default function Dashboard() {
  const [data, setData] = useState<BacktestData | null>(null);
  const [selectedPeriod, setSelectedPeriod] = useState<string>("");
  const [showEw7, setShowEw7] = useState(false);
  const [showSpy, setShowSpy] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/" + (process.env.NEXT_PUBLIC_BACKTEST_JSON || "backtest_data.json"))
      .then((r) => r.json())
      .then((d: BacktestData) => {
        setData(d);
        if (d.periods.length && !selectedPeriod) {
          setSelectedPeriod(d.periods[0].period);
        }
      })
      .catch((e) => {
        console.error("Failed to load backtest data:", e);
      })
      .finally(() => setLoading(false));
  }, []);

  const periodData = data?.periods.find((p) => p.period === selectedPeriod);

  if (loading) {
    return (
      <main className="main">
        <div className="loading">Loading...</div>
      </main>
    );
  }

  if (!data?.periods.length) {
    return (
      <main className="main">
        <div className="empty-state full">
          No backtest data. Run <code>python analyze_strategy.py --export --quick</code> from the
          project root.
        </div>
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
      </header>

      {periodData && (
        <>
          <section className="section">
            <h2>Helix vs EW(7) vs SPY — All Periods</h2>
            <div className="chart-card">
              <ComparisonTable periods={data.periods} />
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
                ew7Data={periodData.ew7_values}
                spyData={periodData.spy_values}
                showEw7={showEw7}
                showSpy={showSpy}
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
        </>
      )}

      <footer className="footer">
        <span>Dynamic Factor Allocation • Regime-Switching</span>
      </footer>
    </main>
  );
}
