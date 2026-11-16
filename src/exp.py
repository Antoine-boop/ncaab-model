"""
Experiment harness.

Protocol (fixed before running experiments):
  * stage 2 is refit OUT-OF-SAMPLE: leave-one-season-out inside 2016-22, expanding window for 2023+.
  * SELECTION uses VAL only:   accuracy on 2016-24, opener edge 2016/2017/2024, close edge 2024,
                               archived-snapshot edge 2018-23.
  * TEST (2025-26 openers/closes, 2027 YTD) is printed only with final=True.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest as BT  # noqa: E402
import model as MD  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")

VAL_OPEN, TEST_OPEN = [2016, 2017, 2024], [2025, 2026]
VAL_CLOSE, TEST_CLOSE = [2024], [2025, 2026]
SNAP = [2018, 2019, 2020, 2021, 2022, 2023]
DEV = list(range(2016, 2023))


def load_base():
    G = pd.read_parquet(os.path.join(PROC, "predictions.parquet"))
    G = G[G.margin.notna() & G.eff_margin.notna() & G.comp_margin.notna() & (G.season >= 2016)].copy()
    G = G.reset_index(drop=True)
    G["Mo_spread"] = BT.market_frame(G, "open").spread.to_numpy()
    G["Mo_total"] = BT.market_frame(G, "open").total.to_numpy()
    G["Mc_spread"] = BT.market_frame(G, "close").spread.to_numpy()
    G["Mc_total"] = BT.market_frame(G, "close").total.to_numpy()
    return G


def design(G, cols):
    X = MD._design(G, [c for c in cols if c in MD.MARGIN_X + MD.TOTAL_X or c == "const"])
    for c in cols:
        if c not in X.columns:
            X[c] = G[c].to_numpy()
    return X[["const"] + [c for c in cols if c != "const"]]


def stage2_oos(G, mcols, tcols, target_m="margin", target_t="total_pts"):
    pm = np.full(len(G), np.nan)
    pt = np.full(len(G), np.nan)
    Xm, Xt = design(G, mcols), design(G, tcols)
    ym, yt = G[target_m].to_numpy(float), G[target_t].to_numpy(float)
    seas = G.season.to_numpy()
    ok = np.isfinite(Xm.to_numpy(float)).all(1) & np.isfinite(Xt.to_numpy(float)).all(1)
    for s in sorted(np.unique(seas)):
        fit = [x for x in DEV if x != s] if s <= 2022 else list(range(2016, s))
        f = np.isin(seas, fit) & ok
        p = (seas == s) & ok
        bm = np.linalg.lstsq(Xm.to_numpy(float)[f], ym[f], rcond=None)[0]
        bt = np.linalg.lstsq(Xt.to_numpy(float)[f], yt[f], rcond=None)[0]
        pm[p] = Xm.to_numpy(float)[p] @ bm
        pt[p] = Xt.to_numpy(float)[p] @ bt
    return pm, pt


def _edge(res, edge):
    m = np.isfinite(res) & np.isfinite(edge)
    if m.sum() < 200:
        return np.nan, np.nan
    r = sm.OLS(res[m], sm.add_constant(edge[m])).fit()
    return r.params[1], r.tvalues[1]


def metrics(G, pm, pt, final=False, roi=True):
    out = {}
    seas = G.season.to_numpy()
    mar, tot = G.margin.to_numpy(float), G.total_pts.to_numpy(float)
    groups = {"dev": DEV, "val": [2023, 2024]} | ({"test": [2025, 2026]} if final else {})
    for g, ss in groups.items():
        m = np.isin(seas, ss) & np.isfinite(pm)
        out[f"mae_m_{g}"] = np.mean(np.abs(mar[m] - pm[m]))
        out[f"mae_t_{g}"] = np.mean(np.abs(tot[m] - pt[m]))
    sets = [("open_val", VAL_OPEN, "Mo"), ("close_val", VAL_CLOSE, "Mc"), ("snap", SNAP, "Mc")]
    if final:
        sets += [("open_test", TEST_OPEN, "Mo"), ("close_test", TEST_CLOSE, "Mc"), ("open_2027", [2027], "Mo")]
    for name, ss, pre in sets:
        m = np.isin(seas, ss)
        sp, tl = G[f"{pre}_spread"].to_numpy(float), G[f"{pre}_total"].to_numpy(float)
        c, t = _edge((mar + sp)[m], (pm + sp)[m])
        out[f"sp_{name}"] = f"{c:.3f}({t:.1f})"
        c, t = _edge((tot - tl)[m], (pt - tl)[m])
        out[f"tot_{name}"] = f"{c:.3f}({t:.1f})"
    if roi:
        out.update(roi_metrics(G, pm, pt, final=final))
    return out


def roi_metrics(G, pm, pt, final=False, thr=0.03):
    H = G.copy()
    H["pred_margin"], H["pred_total"] = pm, pt
    s2sig = 11.3
    from scipy.stats import norm
    H["p_home"] = norm.cdf(pm / s2sig)
    H = H[np.isfinite(pm)]
    M = BT.market_frame(H, "open")
    pools = [2016, 2017, 2024, 2025, 2026]
    out = {}
    tests = [2024] + ([2025, 2026, 2027] if final else [])
    bets = []
    for S in tests:
        fit = [s for s in pools if s < S]
        cal = BT.fit_calibration(H, M, 11.3, 16.9, H.season.isin(fit))
        h = H[H.season == S]
        B = BT.make_bets(h, M.loc[h.index], cal, 11.3, 16.9)
        B.loc[(B.market == "spread") & ~(B.edge_pts >= 1.5), "ev"] = -np.inf
        B.loc[(B.market == "total") & ~(B.edge_pts >= 2.0), "ev"] = -np.inf
        bets.append(B[B.ev >= thr])
    B = pd.concat(bets)
    for (mk, s), b in B[B.market != "ml"].groupby(["market", "season"]):
        out[f"roi_{mk[:2]}_{s}"] = f"{b.profit.mean():+.3f}/{len(b)}"
    if final:
        for mk in ("spread", "total"):
            b = B[(B.market == mk) & B.season.isin([2025, 2026])]
            out[f"roi_{mk[:2]}_test"] = f"{b.profit.mean():+.3f}/{len(b)} z={b.profit.mean() / b.profit.std() * np.sqrt(len(b)):.2f}"
    return out


def run(G, mcols, tcols, label, final=False, roi=True, show=True):
    pm, pt = stage2_oos(G, mcols, tcols)
    r = dict(exp=label) | metrics(G, pm, pt, final=final, roi=roi)
    if show:
        print(pd.Series(r).to_string(), flush=True)
    return r, pm, pt


BASE_M = list(MD.MARGIN_X)
BASE_T = list(MD.TOTAL_X)

if __name__ == "__main__":
    G = load_base()
    run(G, BASE_M, BASE_T, "baseline")
