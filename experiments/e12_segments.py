"""E12: where is the edge reliable? Edge coefficient by segment on VAL openers (2016/17/2024) + snapshots (2018-23)."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd, statsmodels.api as sm
import exp as E
from common import build, M_KEEP, T_KEEP
G = build(); pm, pt = E.stage2_oos(G, M_KEEP, T_KEEP); G["pm"], G["pt"] = pm, pt
HM = {"2", "4", "7", "8", "21", "23"}
G["tier"] = np.select([G.home_conf.isin(HM) & G.away_conf.isin(HM), G.home_conf.isin(HM) | G.away_conf.isin(HM)], ["HM-HM", "HM-other"], "low-low")
G["mon"] = G.date.dt.month.map({11: "1Nov", 12: "2Dec", 1: "3Jan", 2: "4Feb", 3: "5Mar", 4: "5Mar"})
G["gtype"] = np.select([G.neutral, G.conf_game], ["neutral", "conf"], "nonconf-home")
def coef(d, res, edge):
    m = np.isfinite(res) & np.isfinite(edge)
    if m.sum() < 150: return "-"
    r = sm.OLS(res[m], sm.add_constant(edge[m])).fit(); return f"{r.params[1]:.2f}({r.tvalues[1]:.1f}) n={m.sum()}"
for pre, ss, lab in (("Mo", E.VAL_OPEN, "VAL openers"), ("Mc", E.SNAP, "snapshots 2018-23")):
    d = G[G.season.isin(ss)].copy()
    d["absp"] = pd.cut(d[f"{pre}_spread"].abs(), [-0.1, 4, 9, 15, 20, 60])
    d["totb"] = pd.cut(d[f"{pre}_total"], [0, 130, 140, 150, 160, 250])
    print(f"\n=== {lab} ===")
    for seg in ("mon", "absp", "totb", "tier", "gtype"):
        rows = {}
        for k, x in d.groupby(seg, observed=True):
            rows[str(k)] = (coef(x, (x.margin + x[f"{pre}_spread"]).to_numpy(), (x.pm + x[f"{pre}_spread"]).to_numpy()),
                            coef(x, (x.total_pts - x[f"{pre}_total"]).to_numpy(), (x.pt - x[f"{pre}_total"]).to_numpy()))
        print(seg.ljust(6), " | ".join(f"{k}: sp {a} / tot {b}" for k, (a, b) in rows.items()))
