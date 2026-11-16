# NCAAB bottom-up betting model

Players → team component ratings → possessions & points → margin / total / win probability, priced
against sportsbook lines. Full methodology and walk-forward backtest: **[reports/REPORT.md](reports/REPORT.md)**.

**Short version (v2):** profitable vs **opening** totals (+8.2% ROI over 5,008 bets, 2024-26, z = 6.1) and
probably vs opening spreads (+5.2% over 2,572 bets, z = 2.8; weaker on the timing-clean subset).
No demonstrated moneyline edge. Nothing beats closing lines. What was tried and what worked:
**[experiments/LOG.md](experiments/LOG.md)**.

## Pipeline

| Step | Script | Notes |
|---|---|---|
| 1. Scrape | `src/scrape_espn.py 2015 ... 2027 [--end YYYY-MM-DD]` | ESPN scoreboards, box scores, archived odds (open/close) |
| 2. Build | `src/build_dataset.py` | possessions, components, consolidated lines → `data/processed/` |
| 3. Tune | `src/tune_components.py`, `src/refine_params.py` | per-component ridge hyperparameters (dev 2016-22) → `reports/best_params.json` |
| 4. Model | `src/model.py` | walk-forward ratings + player layer + stage 2 → `data/processed/predictions.parquet` |
| 5. Evaluate | `src/evaluate.py` | walk-forward betting backtest → `reports/backtest_summary.csv`, `reports/calibration.json` |
| 6. Daily | `src/predict_today.py [--date] [--no-refresh] [--min-ev 0.03] [--allow-mismatch]` | today's bets vs opening lines |

Supporting modules: `src/features.py` (market-derived ratings, team home edge, conference signals, recent form),
`src/context.py` (rest, back-to-backs, travel, time zones, altitude), `src/exp.py` (experiment harness: out-of-sample
stage 2, validation-only metrics), `experiments/` (one script per experiment + `LOG.md`).

## Setup from a fresh clone

`data/` is not committed (≈110 MB, regenerable). The tuned hyperparameters (`reports/best_params.json`) and
the betting calibration (`reports/calibration.json`) are committed, so tuning is optional.

```bash
pip install -r requirements.txt
python src/scrape_espn.py 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026   # ~25 min
python src/scrape_espn.py 2027 --refresh --end $(date -d yesterday +%F)               # current season
python src/build_dataset.py
python src/model.py          # walk-forward ratings + features + stage 2 (~4 min)
python src/evaluate.py       # backtest + calibration.json
python src/predict_today.py  # today's bets
```

Optional re-tune (≈30 min on 4 cores): `python src/tune_components.py && python src/refine_params.py`.

Set `OPENBLAS_NUM_THREADS=1` (the scripts do). Multithreaded BLAS is slower on these small solves.
