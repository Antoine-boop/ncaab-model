"""
Final walk-forward betting evaluation.

  OPEN lines : test seasons 2024, 2025, 2026, 2027(YTD); calibration fit on all earlier opener seasons
               (2016-17 consensus openers + earlier test seasons).
  CLOSE lines: test seasons 2024-2027 with explicit book close fields; calibration fit on 2016-2022 +
               earlier test seasons.
Also: CLV for opener bets, a timing-clean subset (both teams idle >= 3 days), threshold grid.
Writes reports/backtest_bets.parquet, reports/backtest_summary.csv, reports/calibration.json
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest as BT  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
REP = os.path.join(ROOT, "reports")

POOLS = {"open": [2016, 2017, 2024, 2025, 2026], "close": list(range(2016, 2023)) + [2024, 2025, 2026]}
TESTS = {"open": [2024, 2025, 2026, 2027], "close": [2024, 2025, 2026, 2027]}
THRESHOLDS = [0.0, 0.02, 0.03, 0.04, 0.06, 0.08]
MIN_EDGE_PTS = {"spread": 1.5, "total": 2.0}   # model must disagree with the number, not just the price


def add_rest(G):
    ga = pd.read_parquet(os.path.join(PROC, "games_all.parquet"))
    ga = ga[ga.completed]
    long = pd.concat([ga[["season", "date", "home_id"]].rename(columns={"home_id": "t"}),
                      ga[["season", "date", "away_id"]].rename(columns={"away_id": "t"})]).drop_duplicates()
    long = long.sort_values(["t", "date"])
    long["prev"] = long.groupby(["season", "t"]).date.shift()
    rest = long.set_index(["season", "t", "date"]).prev
    for s, col in (("h", "home_id"), ("a", "away_id")):
        prev = rest.reindex(pd.MultiIndex.from_arrays([G.season, G[col], G.date])).to_numpy()
        G[f"{s}_rest"] = (G.date - pd.to_datetime(prev)).dt.days
    G["min_rest"] = G[["h_rest", "a_rest"]].min(axis=1)
    return G


def run():
    G = pd.read_parquet(os.path.join(PROC, "predictions.parquet"))
    s2 = pd.read_pickle(os.path.join(PROC, "stage2.pkl"))
    sig_m, sig_t = s2["sig_m"], s2["sig_t"]
    G = G[G.margin.notna() & G.pred_margin.notna()].copy()
    G = add_rest(G)
    Mo, Mc = BT.market_frame(G, "open"), BT.market_frame(G, "close")
    allb = []
    cals = {}
    for lt, M in (("open", Mo), ("close", Mc)):
        for S in TESTS[lt]:
            fit = [s for s in POOLS[lt] if s < S]
            cal = BT.fit_calibration(G, M, sig_m, sig_t, G.season.isin(fit))
            g = G[G.season == S]
            B = BT.make_bets(g, M.loc[g.index], cal, sig_m, sig_t)
            B["line_type"] = lt
            allb.append(B)
        cals[lt] = BT.fit_calibration(G, M, sig_m, sig_t, G.season.isin(POOLS[lt]))
    B = pd.concat(allb)
    for mk, mn in MIN_EDGE_PTS.items():
        B.loc[(B.market == mk) & ~(B.edge_pts >= mn), "ev"] = -np.inf
    # attach CLV / context
    ctx = G[["game_id", "min_rest", "margin", "total_pts"]].copy()
    ctx["mo_spread"], ctx["mc_spread"] = Mo.spread.to_numpy(), Mc.spread.to_numpy()
    ctx["mo_total"], ctx["mc_total"] = Mo.total.to_numpy(), Mc.total.to_numpy()
    B = B.merge(ctx, on="game_id", how="left")
    B["clv_pts"] = np.where(B.market == "spread", (B.mo_spread - B.mc_spread) * B.side,
                            np.where(B.market == "total", (B.mc_total - B.mo_total) * B.side, np.nan))
    B["month"] = pd.to_datetime(B.date).dt.month

    rows = []
    for (lt, mk), b0 in B.groupby(["line_type", "market"]):
        for thr in THRESHOLDS:
            b1 = b0[b0.ev >= thr]
            for S, b in list(b1.groupby("season")) + [("2024-26", b1[b1.season.isin([2024, 2025, 2026])]),
                                                      ("2024-26 clean", b1[b1.season.isin([2024, 2025, 2026])
                                                                         & (b1.min_rest >= 3)])]:
                n = len(b)
                if n == 0:
                    continue
                rows.append(dict(line_type=lt, market=mk, thr=thr, season=S, n=n,
                                 win=round((b.profit > 0).mean(), 4), roi=round(b.profit.sum() / n, 4),
                                 units=round(b.profit.sum(), 1),
                                 z=round(b.profit.mean() / (b.profit.std(ddof=1) / np.sqrt(n)), 2) if n > 2 else np.nan,
                                 clv=round(b.clv_pts.mean(), 2) if mk != "ml" else np.nan,
                                 clv_pos=round((b.clv_pts > 0).mean(), 3) if mk != "ml" else np.nan))
    R = pd.DataFrame(rows)
    os.makedirs(REP, exist_ok=True)
    B.to_parquet(os.path.join(REP, "backtest_bets.parquet"), index=False)
    R.to_csv(os.path.join(REP, "backtest_summary.csv"), index=False)
    cj = {k: {m: np.asarray(v[m]).tolist() for m in ("spread", "total", "ml_kappa")} for k, v in cals.items()}
    cj["current"] = cj["close"]           # moved lines: price conservatively with closing-line calibration
    cj["sig_m"], cj["sig_t"] = sig_m, sig_t
    with open(os.path.join(REP, "calibration.json"), "w") as fh:
        json.dump(cj, fh, indent=1)
    return R, B


if __name__ == "__main__":
    R, B = run()
    pd.set_option("display.width", 200)
    for lt in ("open", "close"):
        print(f"\n===== {lt.upper()} lines =====")
        x = R[(R.line_type == lt) & R.thr.isin([0.03])]
        print(x.drop(columns=["line_type"]).to_string(index=False))
    print("\n===== threshold grid, pooled 2024-26 =====")
    x = R[R.season.isin(["2024-26", "2024-26 clean"])]
    print(x.pivot_table(index=["line_type", "market", "season"], columns="thr", values="roi").round(3).to_string())
