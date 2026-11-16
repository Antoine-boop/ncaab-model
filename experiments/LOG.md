# Experiment log

Scripts in this folder are the record of what was tried. Run them from the repo root after
`python src/model.py`; some read intermediates from `experiments/.cache/` (git-ignored) written by earlier
experiments (e.g. E01 writes the market ratings used by E02+). The validated features are reimplemented in
`src/features.py` / `src/context.py`, so the production numbers come from `src/model.py` + `src/evaluate.py`.

Protocol: stage 2 refit out-of-sample (LOSO 2016-22, expanding 2023+). Selection on VAL only:
dev/val MAE, edge coefficient (t) vs openers 2016/2017/2024, vs archived snapshots 2018-23.
TEST (2025-26, 2027 YTD) is evaluated once at the end.

Baseline: MAE m dev 8.887 / val 8.946; sp_open_val 0.280(7.1); tot_open_val 0.572(13.0); sp_snap 0.308(12.9); tot_snap 0.388(12.2)

| # | Idea | Result (VAL) | Verdict |
|---|---|---|---|
| E01 | Market-derived ratings from previous games' closing lines (implied points ridge, lam=0.3 g, no decay, rho 0.7) | sp_open_val 0.280->0.376 (t 8.9); sp_snap t 12.9->14.1; MAE m dev 8.887->8.860, val 8.946->8.908; totals ~flat (0.572->0.593) | KEEP for sides (weight ~0.27) |
| E02 | Mismatch nonlinearity in stage 2 (signed-square, cubic, abs-margin for totals) | sp_open_val +0.006, MAE flat/worse | REJECT |
| E03 | Multiplicative (log-PPP) efficiency instead of / on top of additive | identical to additive (0.343 vs 0.343); +0.004 when stacked | REJECT. (Side result: roster priors beat team-only priors: 0.359 vs 0.343, MAE dev -0.020) |
| E04 | Schedule/travel context: back-to-back, rest, games-in-7-days, travel miles, time zones, venue altitude | sp_open_val 0.359->0.386 (t 9.4), sp_snap 13.8->14.4, MAE m val -0.012; tot_open_val 0.586->0.599 (t 13.4), MAE t val -0.009. b2b_diff -2.5 pts (t -5.3), tz +0.28/zone (t 2.6), altitude +0.29 total pts/1000 ft (t 3.6) | KEEP |
| E05 | Availability with empirical return probabilities (missed 1 game -> 48% play next, 2 -> 35%, 3 -> 23%, 6+ -> 5%) instead of "missed last game = out" | sp_open_val 0.386->0.384, MAE m val +0.003; totals +0.001 | REJECT (keep simpler rule; market injury reports resolve the uncertain cases) |
| E06 | Team-specific home edge (home-vs-road residual split, prior 3 seasons, shrunk k=40) + arena scoring effect for totals | sides: sp_open_val 0.386->0.402 (t 9.8), sp_snap 14.4->15.0, MAE m dev -0.005 val -0.009. totals: tot_snap 12.4->12.8, MAE t val -0.004, open flat | KEEP both |
| E07 | Conference signals: (a) conference residual in non-conference games (sides), (b) conference scoring environment in league games (totals); walk-forward within season, shrunk k=30 | sides: sp_open_val 0.402->0.417 (t 10.2), sp_snap 15.0->15.3, MAE m dev -0.004. totals: MAE t val -0.015, tot_snap 12.8->13.3, open t 13.4->13.5 | KEEP both |
| E08 | Recent form: team's mean residual (vs walk-forward efficiency projection) over recent games, sides + totals | ewm half-life 6: sp_open_val 0.417->0.489 (t 11.3), sp_snap 15.3->16.0, MAE m val -0.022; tot_open_val 0.601->0.638 (t 13.8), MAE t val -0.009. Season-to-date mean gets most of it (rating-bias correction) | KEEP (ewm6) |
| E08b | Pace interactions for totals (tempo x efficiency, tempo^2) | no change | REJECT |
| E09 | LightGBM on the residual of the linear stage 2 (huber, 15 leaves, 150 rounds; same OOS folds) | sp_open_val 0.489->0.506, tot_open 0.638->0.656, MAE val -0.014/-0.006; but sp_snap 16.0->15.8; 400 rounds overfits. Importance diffuse (no single interaction) | PARK (optional; small, mixed) |
| E10 | Referee crew effects on totals | ESPN lists officials only after tip-off (0 of 12 upcoming games) | SKIP (not actionable pre-game) |
| E11 | Early-season / November interactions (market rating, form, conference, availability) | coefficients shift (market weight 0.27 -> 0.57 early) but t-stats/MAE ~unchanged | REJECT (no measurable gain) |
| E12 | Segment reliability of edge (VAL openers + 2018-23 snapshots) | Edge weak in: non-conference home games, high-major vs smaller school, |spread|>20, November totals. Strongest in low-major vs low-major and conference games. Consistent across both VAL sets | -> segment-aware calibration (E13) |
| E13 | Segment-aware calibration (z x non-conf-home / mismatch / November / low-major) | OOS log-loss on 2024 openers: spreads 690.43->689.89 (trivial), totals 689.54->690.94 (worse); interaction estimates unstable across fits | REJECT. Adopt only the mismatch exclusion (|spread|>20), now supported by VAL (openers + snapshots) |
| E14 | Recency-weighted stage-2 fit (0.85 / 0.7 per season back) | no change (+-0.002) | REJECT |
