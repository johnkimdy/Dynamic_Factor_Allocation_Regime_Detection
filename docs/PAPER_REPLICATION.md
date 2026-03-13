# Paper Replication Workflow

Replicate Princeton paper (arXiv:2410.14841) test period and methodology.

## 1. Tune with paper-aligned config

```bash
python tune_sjm_hyperparameters.py --config hyperparam/paper_aligned.json
```

- **Validation**: 2007-01-01 to 2022-12-31 (paper test period start)
- **Holdout**: 2023-01-01 to 2024-12-31
- **Min train**: 8 years (2016 days) per paper
- Output: `hyperparam/sjm_hyperparameters_YYYYMMDD_HHMMSS.json`

## 2. Run analysis with paper params

```bash
# Use the tune output from step 1 (replace with actual filename)
python analyze_strategy.py --config hyperparam/sjm_hyperparameters_YYYYMMDD_HHMMSS.json --target-te 0.02
```

- `--config`: Load SJM params from paper-aligned tune run
- `--target-te 0.02`: Target 2% tracking error (paper: 1-4%)

## 3. Paper test periods

Analysis includes 2007-2024, 2007-2023, 2007-2022 (paper Exhibit 1 timeframe).

## Remaining differences

| Item | Paper | Status |
|------|-------|--------|
| Nystrup lookback algorithm | Online inference with lookback | Not implemented |
| Rebalance trigger | Apply BL weights at T+2 | 2% threshold |
