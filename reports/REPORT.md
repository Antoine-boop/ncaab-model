# NCAAB bottom-up model: build notes & backtest report (v2)

_Data through 2026-11-15. All betting results are **walk-forward**: no parameter, blend weight, calibration
or threshold was fit on the season it is scored on. v2 features were **selected on validation data only**
(2016-24); 2025-26 was scored once, after selection._

## Bottom line

| Market | vs **opening** line, 2024-26, EV ≥ 3% | vs **closing** line | Verdict |
|---|---|---|---|
| **Totals** | **+8.2% ROI**, 5,008 bets, 56.7% win, z = 6.1. Every season positive (+7.4 / +3.8 / +12.5%). 66% of bets beat the close (avg CLV +1.5 pts) | −3.6% | **Profitable at openers. Core product.** |
| **Spreads** | **+5.2% ROI**, 2,572 bets, z = 2.8. Every season positive (+6.6 / +4.5 / +5.2%) | −2.0% | **Probably profitable at openers.** Timing-clean subset weaker (+3.0%, z = 1.3) |
| **Moneyline** | −1.3% (+3.2% at EV ≥ 8%, not significant) | −3.6% | **No demonstrated edge.** |

* **Nothing beats NCAAB closing lines, and this model doesn't either.** The edge is the gap between opening
  lines and what box scores, schedules and the market's own recent history already say. Bet **early, at or
  near the opener**. Openers carry low limits.
* **Current season (Nov 1-15, 2026) is weak:** totals +1.9% (236 bets), spreads −10.1% (214 bets, about −1.5σ).
  November is the weakest month. The daily script requires EV ≥ 6% for November spreads.

## v2: what was tried, what worked

Full log with numbers: [`experiments/LOG.md`](../experiments/LOG.md). Harness: `src/exp.py`. Stage 2 is refit
out-of-sample (leave-one-season-out 2016-22, expanding window after). Validation = dev/val accuracy + edge vs
2016/2017/2024 openers + edge vs 2018-23 archived lines.

| Idea | Validation effect | Verdict |
|---|---|---|
| **Market-derived ratings** (ridge on previous games' closing lines) | spread edge 0.28 → 0.38, MAE −0.03 | **kept** |
| **Schedule/travel**: back-to-back (−2.5 pts, t −5.3), time zones, travel, altitude (+0.3 total pts / 1,000 ft) | spread t 8.7 → 9.4, total t 13.0 → 13.4 | **kept** |
| **Team-specific home edge** + **arena scoring** (prior 3 seasons, shrunk) | spread t → 9.8; total snapshots t → 12.8 | **kept** |
| **Conference signals** (non-conference over/under-performance; league scoring environment) | spread t → 10.2; total MAE −0.015 | **kept** |
| **Recent form** (exp-weighted residuals, half-life 6 games) | spread 0.42 → 0.49 (t 11.3); total 0.60 → 0.64 | **kept** (largest single gain) |
| **Mismatch exclusion** (|spread| ≥ 20) | edge weak there on both validation sets | **kept** (bet rule) |
| Nonlinear / multiplicative efficiency, early-season interactions, recency-weighted fits | no measurable change | rejected |
| Injury return probabilities (48% return after 1 missed game, 35% after 2...) | no gain over "missed last game = out" | rejected |
| Segment-aware calibration | worse out-of-sample log-loss for totals; unstable | rejected |
| Gradient-boosted stage 2 | small, mixed gain; overfits fast | parked |
| Referee crews | not published before tip-off | skipped |

Validation, baseline → v2: spread edge vs openers **0.28 → 0.50** (t 7.1 → 11.4), total edge **0.57 → 0.65**
(t 13.0 → 13.9), MAE margin 8.95 → 8.87, total 12.98 → 12.94.

## Architecture

```
players (box scores)                              teams (box scores)
  │ player value = team net rating                   │ per-component opponent-adjusted ridge ratings
  │   + game-score deviation vs teammates            │   (walk-forward, recency decay, preseason priors)
  ▼                                                  │  ppp, tempo, TO%, OREB/miss, FTA/FGA,
roster priors ─────► team priors ──────────────────► │  FT%, 3PA/FGA, 3P%, 2P%
availability (who missed the last game)               ▼
                     components ──(exact possession identity)──► points per possession × tempo
                                                      ▼
  + market-derived ratings, schedule/travel/altitude, team home edge, conference signals, recent form
                                                      ▼
            stage 2 (linear blend) ──► margin, total, P(home win)
                                                      ▼
            calibration vs opening lines (walk-forward) ──► P(cover), EV at the real price
```

1. **Component ratings** (`src/ratings.py`). Each stat gets its own ridge model,
   `y = μ + off[team] + def[opp] + hca·loc`, with prior strength, recency and carry-over tuned per component
   on 2016-22. Opponent 3P% and FT% are shrunk heavily (mostly luck). 3PA rate is learned fast.
2. **Possession identity** (`ppp_from_components`). Rebuilds points per possession exactly from the components.
3. **Player layer** (`src/players.py`, `src/priors.py`): roster-based priors (prior-season player values
   follow transfers) and availability adjustments.
4. **Engineered features** (`src/features.py`, `src/context.py`): the v2 additions above, all computed from
   pre-tip information.
5. **Stage 2** (`src/model.py`). Linear blend. Backtests use out-of-sample fits; live picks use a fit on every
   completed season (2016-26).
6. **Pricing** (`src/backtest.py`). Logistic calibration of `(model − line) / σ(line)`, with σ growing in
   mismatches and with the total. Bets need EV ≥ 3% **and** a real point disagreement (≥ 1.5 spread, ≥ 2 total),
   and no mismatch (|spread| ≥ 20).

## Data

ESPN public APIs (`src/scrape_espn.py`): 65,400 D1-vs-D1 games 2014-15 → 2026-27, team and player box scores,
and every book ESPN archived. Explicit open + close exist from 2023-24 (2024: ~11 books; 2025: ESPN BET;
2026-27: DraftKings). 2015-17 have a consensus "Opening" feed. 2018-23 lines are archived snapshots of unknown
timing, used only for validation, never for test claims.

## Results vs opening lines (v2, EV ≥ 3%)

| Season | Totals n | Totals ROI | Spreads n | Spreads ROI |
|---|---|---|---|---|
| 2023-24 | 1,282 | +7.4% | 510 | +6.6% |
| 2024-25 | 1,743 | +3.8% | 1,026 | +4.5% |
| 2025-26 | 1,983 | +12.5% | 1,036 | +5.2% |
| **2024-26 pooled** | **5,008** | **+8.2% (z 6.1)** | **2,572** | **+5.2% (z 2.8)** |
| 2024-26 timing-clean¹ | 3,670 | +8.0% (z 5.2) | 1,774 | +3.0% (z 1.3) |
| 2026-27 YTD | 236 | +1.9% | 214 | −10.1% |

¹ Both teams idle ≥ 3 days, so the opener could not have pre-dated a result the model used.

* **By month (2024-26):** totals Nov +5.3%, Dec +8.2%, Jan +7.4%, Feb +10.0%, Mar +10.8%.
  Spreads Nov +2.5%, Dec +8.9%, Jan +4.1%, Feb +5.2%, Mar +13.6%.
* Unders +10.2%, overs +7.5%.
* **Threshold grid (2024-26):**

| min EV | Totals n | Totals ROI | Spreads n | Spreads ROI |
|---|---|---|---|---|
| 0% | 7,113 | +6.8% | 4,612 | +3.4% |
| 2% | 5,826 | +7.7% | 3,200 | +4.4% |
| 3% | 5,008 | +8.2% | 2,572 | +5.2% |
| 4% | 4,280 | +7.4% | 2,070 | +5.7% |
| 6% | 2,896 | +9.8% | 1,329 | +6.9% |
| 8% | 1,876 | +11.1% | 813 | +8.2% |

### Model accuracy (MAE, points): model / open / close

| Season | Margin | Total |
|---|---|---|
| 2023-24 | 8.88 / 8.88 / 8.84 | 12.97 / 13.01 / 12.90 |
| 2024-25 | 8.91 / 8.90 / 8.79 | 12.79 / 12.81 / 12.66 |
| 2025-26 | 9.06 / 9.07 / 8.94 | 13.53 / 13.62 / 13.38 |

v2 matches the opener on margins and beats it on totals in every season. The close remains better.

## Known weaknesses / risk

* **Early season.** The current season is flat-to-negative so far. Nov spreads are the weakest segment.
* **Spread timing.** The spread edge shrinks on the timing-clean subset, so stake spreads smaller than totals.
* **Mismatches (|spread| ≥ 20)** are excluded. The model under-projects big favourites and the market prices
  these games well (both validation sets agree).
* **Variance.** Same-night totals are correlated. In v1, quarter-Kelly capped at 2%/bet still had 25-57%
  drawdowns within a season. The live script uses quarter-Kelly capped at **1%**. Also cap daily exposure (e.g. 10%).
* **Execution.** Edges are measured at DraftKings/ESPN BET openers. If the number has moved against you, pass.
* **No recruiting or injury-report data.** Freshmen get an average prior, and an injured player counts as out
  only after he misses a game.

## Daily use

```bash
python src/predict_today.py                # refresh data, project today's slate, price vs openers
python src/predict_today.py --date 2026-11-17 --no-refresh
```
Output: `reports/picks_<date>.csv`, all projections plus a filtered bet list (EV thresholds, point-edge
minimums, mismatch filter, quarter-Kelly stake %, and whether the opening number is still available).
