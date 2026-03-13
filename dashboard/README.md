# Helix Factor Strategy Dashboard

A Next.js app that visualizes active rebalancing across different backtest periods.

## Quick Start

```bash
# From project root
cd dashboard
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Multiple Backtest Configs

Use different JSON files for different backtest runs (e.g. different hyperparams):

```bash
# Load a specific backtest file (must exist in public/)
npm run dev -- --config backtest_data_v2.json
npm run build -- --config backtest_data_v2.json
```

## Data

The dashboard reads `public/backtest_data.json` by default.

To generate backtest data from the Python strategy:

```bash
# From project root (activate your conda/venv with deps first)
python analyze_strategy.py --export --quick   # 3 periods, ~2–5 min
python analyze_strategy.py --export           # All periods, ~15–30 min
```

**Export with a specific hyperparam config:**

```bash
# Use a different SJM hyperparam JSON (-o defaults to backtest_data_<datetime>.json when -c given)
python analyze_strategy.py -c hyperparam/sjm_hyperparameters_20260313_051222.json --export
python analyze_strategy.py -c hyperparam/sjm_hyperparameters_v2.json --export -o dashboard/public/backtest_data_v2.json
```

## Deploy to Vercel

Deploy to **dfardjump.vercel.app** (or any custom name):

```bash
# From dashboard folder
cd dashboard
npx vercel
```

When prompted for **project name**, enter `dfardjump` to get `https://dfardjump.vercel.app`.

For production: `npx vercel --prod`

**Note:** `public/backtest_data.json` (and any `--config` file) must exist locally. For Git-based deploys, commit the JSON files; for CLI deploy, they’re included from your working directory.

## Features

- **Period dropdown** – Switch between backtest windows (e.g. 2024–2025, 2022–2024)
- **Performance metrics** – Total return, Sharpe, volatility, max drawdown, rebalance count
- **Portfolio value chart** – Cumulative growth (normalized to 100)
- **Allocation chart** – Factor weights at each rebalance date (QUAL, MTUM, USMV, VLUE, SIZE, IWF)
