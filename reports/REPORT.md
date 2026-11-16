# NCAAB bottom-up model: build notes & backtest report

_Data through 2026-11-15. All betting results below are **walk-forward**: no parameter, blend weight,
calibration or threshold was fit on the season it is scored on._

## Bottom line

| Market | vs **opening** line (2024-26, EV ≥ 3%) | vs **closing** line | Verdict |
|---|---|---|---|
| **Totals** | **+6.0% ROI**, 5,010 bets, 55.5% win, z = 4.5. Positive every season (+5.7 / +2.4 / +9.3%). 63% of bets beat the close (avg CLV +1.4 pts) | −2.7% | **Profitable at openers. Core product.** |
| **Spreads** | **+6.4% ROI**, 1,338 bets, 54.9% win, z = 2.4 | −6.3% | **Probably profitable at openers.** Weaker evidence (see timing check) |
| **Moneyline** | −4.9% ROI | −8.4% | **No edge found. Don't bet ML off this model.** |

* **Nobody's model beats NCAAB closing lines, and this one doesn't either.** The edge is the gap between
  opening lines and what the box scores already say. You only capture it by betting **early, at or near
  the opener**. Openers carry low limits.
* **The current season (2026-27, Nov 1-15) is flat:** totals +1.2% (293 bets), spreads −3.4% (225 bets).
  November is the model's weakest month (only last year's data and roster info are available).

## What "bottom-up" means here

```
players (box scores)                              teams (box scores)
  │ player value = team net rating                   │ per-component opponent-adjusted ratings
  │   + game-score deviation vs teammates            │   (walk-forward ridge, recency decay,
  ▼                                                  │    preseason priors)
roster priors ──────────────► team priors ─────────► │  ppp, tempo, TO%, OREB/miss, FTA/FGA,
availability (who missed the last game)               │  FT%, 3PA/FGA, 3P%, 2P%
                                                      ▼
                    components ──(exact possession identity)──► points per possession × tempo
                                                      ▼
            stage 2 (linear blend, fit on 2016-22 only) ──► margin, total, P(home win)
                                                      ▼
            calibration vs opening lines (walk-forward) ──► P(cover), EV at the real price
```

1. **Component ratings** (`src/ratings.py`). Each stat gets its own ridge model,
   `y = μ + off[team] + def[opp] + hca·loc`. Possession-weighted, exponential recency decay, shrunk toward
   a preseason prior. Exponential decay lets the normal equations update incrementally per day, and a Schur
   complement on the diagonal offense block makes the daily refits fast. Prior strength, recency and
   carry-over are tuned **per component** on 2016-22 (`reports/tune_*.csv`). The data's choices make
   basketball sense:
   * 3PA rate: light shrinkage (3 games), 60-day recency. Shot profile is a stable, fast-learned identity.
   * 3P% allowed: 100-game shrinkage. Opponent FT% allowed: 100 games. Both are mostly luck.
   * TO%, OREB, 2P%: ~10 games. Tempo: 30-day recency. Efficiency: no decay, 0.92 carry-over.
2. **Possession identity** (`ppp_from_components`). Rebuilds points per possession exactly (error 3e-5)
   from TO%, OREB/miss, FTA/FGA, FT%, 3PA/FGA, 3P%, 2P%. That lets luck-prone pieces be shrunk without
   distorting the stable ones.
3. **Player layer** (`src/players.py`, `src/priors.py`):
   * *Roster priors*: minutes-weighted prior-season player values, carried across transfers, for players
     already seen in this season's box scores. Out-of-sample this cut preseason team-rating error ~8%.
     Players with no prior D1 data are clearly below average.
   * *Availability*: expected lineup (players who missed the last game assumed out) minus the lineup the
     rating was fit on. Market check: the **closing** spread under-reacts to these absences
     (coef +0.67, t = 2.9 on 2024-26).
4. **Stage 2** (`src/model.py`). Linear blend fit on 2016-22 only. Inputs: efficiency and component
   projections, pace-scaled component terms, availability, early-season interactions, expected overtime
   points, neutral site, conference game, postseason.
5. **Pricing** (`src/backtest.py`). Logistic calibration of `(model − line) / σ` → P(cover). σ depends
   on the game: margin σ grows from 11.0 to 13+ in mismatches, total σ grows with the total. Bets need
   EV ≥ 3% **and** a real point disagreement (≥ 1.5 spread, ≥ 2 total), so you never bet a stale price.

## Data

ESPN public APIs (`src/scrape_espn.py`): 65,400 D1-vs-D1 games 2014-15 → 2026-27, team and player box
scores, and every book ESPN archived. **Explicit open + close for spread/total/ML exist for 2023-24
onward** (2024: ~11 books; 2025: ESPN BET; 2026-27: DraftKings). 2015-17 have a consensus "Opening" feed.
2018-23 lines are archived snapshots of unknown timing, so they're not used for test claims.

## Validation protocol

* Model hyperparameters, roster-prior regressions, stage-2 weights, σ functions: **2016-22 only**.
* Opening-line calibration for season S: fit on all earlier opener seasons only
  (2016-17, then 2024, 2025, ...). Closing-line test: explicit-close seasons only.
* Bets settle at the archived price for that book (juice included; −110 only if missing).
* **Timing check.** If an opener is posted before a team's previous game ends, the model could "know" a
  result the opener didn't. The "clean" rows keep only games where both teams were idle ≥ 3 days.

## Results vs opening lines (EV ≥ 3%)

| Season | Totals n | Totals ROI | Spreads n | Spreads ROI |
|---|---|---|---|---|
| 2023-24 | 1,255 | +5.7% | 201 | +0.5% |
| 2024-25 | 1,722 | +2.4% | 602 | +8.8% |
| 2025-26 | 2,033 | +9.3% | 535 | +6.0% |
| **2024-26 pooled** | **5,010** | **+6.0% (z 4.5)** | **1,338** | **+6.4% (z 2.4)** |
| 2024-26 timing-clean | 3,602 | +5.8% (z 3.7) | 906 | +3.2% (z 1.0) |
| 2026-27 YTD | 293 | +1.2% | 225 | −3.4% |

* **Totals hold up on the timing-clean subset. Spreads mostly don't** (z 2.4 → 1.0), so treat the spread
  edge as unproven and stake it smaller.
* Over bets +5.7%, under bets +6.7%. The edge isn't just "scoring went up and openers lagged."
* **By month (2024-26):** totals Nov +1.6%, Dec +3.4%, Jan +6.0%, Feb +12.5%, Mar +7.3%.
  Spreads Nov +1.8%, Dec +9.1%, Jan +11.2%, Feb +15.3%. Every month positive; the edge grows as the
  season's data accumulates.
* **Threshold grid (2024-26):**

| min EV | Totals n | Totals ROI | Spreads n | Spreads ROI | ML ROI |
|---|---|---|---|---|---|
| 0% | 7,624 | +3.6% | 3,315 | +2.1% | −7.2% |
| 2% | 5,942 | +5.4% | 1,877 | +3.6% | −6.0% |
| 3% | 5,010 | +6.0% | 1,338 | +6.4% | −4.9% |
| 4% | 4,141 | +6.5% | 987 | +6.6% | −2.8% |
| 6% | 2,648 | +7.7% | 509 | +8.7% | +0.8% |

### Model accuracy (MAE, points)

| Season | Margin: model / open / close | Total: model / open / close |
|---|---|---|
| 2023-24 | 8.96 / 8.88 / 8.84 | 13.01 / 13.01 / 12.90 |
| 2024-25 | 8.99 / 8.90 / 8.79 | 12.83 / 12.81 / 12.66 |
| 2025-26 | 9.11 / 9.07 / 8.94 | 13.60 / 13.62 / 13.38 |

The model is about as accurate as the opener on totals and slightly worse on sides. It pays because its
errors are partly **independent** of the market's: regressing (result − open) on (model − open) gives
+0.54 for totals (t = 12.7) and +0.34 for spreads (t = 10.2) on 2024-26.

## Known weaknesses / risk

* **Mismatches (|spread| ≥ 20).** The model under-projects big favourites. In 2024-26, November
  underdogs ≥ 20 went 23-35 (39.7%, −23.9% ROI). The live script skips these by default (`--allow-mismatch` to
  override). The filter was found on test data, so the headline numbers above **exclude** it. With it:
  totals +6.7%, spreads +7.7%.
* **Variance.** Same-night totals are correlated. A quarter-Kelly simulation capped at 2%/bet still had
  25-57% drawdowns within a season. The live script uses quarter-Kelly capped at **1%** per bet. Also cap
  daily exposure (e.g. 10%).
* **Execution.** The edge is measured at DraftKings/ESPN BET openers. If the line has moved against you,
  most of the edge is gone. Pass (`available_now = False`).
* **No recruiting/injury-report data.** Freshmen are an average prior, and an injured player is assumed
  out only after he misses a game.

## Daily use

```bash
python src/predict_today.py                # refresh data, project today's slate, price vs openers
python src/predict_today.py --date 2026-11-17 --no-refresh
```
Output: `reports/picks_<date>.csv`, all projections plus a filtered bet list (EV ≥ 3%, point-edge minimums,
mismatch filter, quarter-Kelly stake %, and whether the opening number is still available).
