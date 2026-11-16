# NCAAB bottom-up betting model

Players → team component ratings → possessions & points → margin / total / win probability, priced
against sportsbook lines. Full methodology and walk-forward backtest: **[reports/REPORT.md](reports/REPORT.md)**.

**Short version:** profitable vs **opening** totals (+6.0% ROI over 5,010 bets, 2024-26, z = 4.5) and
probably vs opening spreads (+6.4%, weaker evidence). No moneyline edge. Nothing beats closing lines.

## Pipeline

| Step | Script | Notes |
|---|---|---|
| 1. Scrape | `src/scrape_espn.py 2015 ... 2027 [--end YYYY-MM-DD]` | ESPN scoreboards, box scores, archived odds (open/close) |
| 2. Build | `src/build_dataset.py` | possessions, components, consolidated lines → `data/processed/` |
| 3. Tune | `src/tune_components.py`, `src/refine_params.py` | per-component ridge hyperparameters (dev 2016-22) → `reports/best_params.json` |
| 4. Model | `src/model.py` | walk-forward ratings + player layer + stage 2 → `data/processed/predictions.parquet` |
| 5. Evaluate | `src/evaluate.py` | walk-forward betting backtest → `reports/backtest_summary.csv`, `reports/calibration.json` |
| 6. Daily | `src/predict_today.py [--date] [--no-refresh] [--min-ev 0.03] [--allow-mismatch]` | today's bets vs opening lines |

## Setup from a fresh clone

`data/` is not committed (≈110 MB, regenerable). The tuned hyperparameters (`reports/best_params.json`) and
the betting calibration (`reports/calibration.json`) are committed, so tuning is optional.

```bash
pip install -r requirements.txt
python src/scrape_espn.py 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026   # ~25 min
python src/scrape_espn.py 2027 --refresh --end $(date -d yesterday +%F)               # current season
python src/build_dataset.py
python src/model.py          # walk-forward ratings + stage 2 (~4 min)
python src/evaluate.py       # backtest + calibration.json
python src/predict_today.py  # today's bets
```

Optional re-tune (≈30 min on 4 cores): `python src/tune_components.py && python src/refine_params.py`.

Set `OPENBLAS_NUM_THREADS=1` (the scripts do). Multithreaded BLAS is slower on these small solves.
