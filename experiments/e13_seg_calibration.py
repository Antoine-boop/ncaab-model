"""E13: segment-aware calibration (z x segment interactions), judged by out-of-sample log-loss + ROI."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd, statsmodels.api as sm
import exp as E, backtest as BT
from common import build, M_KEEP, T_KEEP
G = build(); pm, pt = E.stage2_oos(G, M_KEEP, T_KEEP); G["pm"], G["pt"] = pm, pt
HM = {"2", "4", "7", "8", "21", "23"}
G["nch"] = (~G.neutral & ~G.conf_game).astype(float)
G["lowlow"] = (~G.home_conf.isin(HM) & ~G.away_conf.isin(HM)).astype(float)
G["nov"] = (G.date.dt.month == 11).astype(float)
def seg_X(d, z, spread, kind):
    mis = np.clip((np.abs(spread) - 12) / 8, 0, 1.5)
    cols = {"c": np.ones(len(d)), "z": z}
    if kind != "plain":
        cols |= {"z_nch": z * d.nch.to_numpy(), "z_mis": z * mis, "z_nov": z * d.nov.to_numpy(), "z_ll": z * d.lowlow.to_numpy()}
    return pd.DataFrame(cols, index=d.index)
def prep(d, pre, market):
    L, T = d[f"{pre}_spread"].to_numpy(float), d[f"{pre}_total"].to_numpy(float)
    if market == "spread":
        res = d.margin.to_numpy() + L; z = (d.pm.to_numpy() + L) / BT.sig_margin(L)
    else:
        res = d.total_pts.to_numpy() - T; z = (d.pt.to_numpy() - T) / BT.sig_total(T)
    ok = np.isfinite(res) & np.isfinite(z) & (res != 0) & np.isfinite(L)
    return d[ok], z[ok], (res[ok] > 0).astype(float), L[ok]
out = []
for market in ("spread", "total"):
    for kind in ("plain", "segment"):
        for fit_sets, test_s, lab in (([("Mo", [2016, 2017])], 2024, "fit openers 16-17"),
                                      ([("Mo", [2016, 2017]), ("Mc", E.SNAP)], 2024, "fit openers+snapshots")):
            Xs, ys = [], []
            for pre, ss in fit_sets:
                d, z, y, L = prep(G[G.season.isin(ss)], pre, market); Xs.append(seg_X(d, z, L, kind)); ys.append(y)
            mdl = sm.Logit(np.concatenate(ys), pd.concat(Xs)).fit(disp=0)
            d, z, y, L = prep(G[G.season == test_s], "Mo", market)
            p = mdl.predict(seg_X(d, z, L, kind)); p = np.clip(p, 1e-4, 1 - 1e-4)
            ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
            # flat -110 ROI for p >= 0.55 side picks
            side_p = np.maximum(p, 1 - p); win = np.where(p >= 0.5, y, 1 - y); b = side_p >= 0.55
            out.append(dict(market=market, calib=kind, fit=lab, test=test_s, logloss_x1000=round(1000 * ll, 2),
                            bets=int(b.sum()), win=round(win[b].mean(), 4), roi_110=round((win[b] * (1 + 100 / 110) - 1).mean(), 4),
                            params=np.round(mdl.params.to_numpy(), 2).tolist()))
print(pd.DataFrame(out).to_string(index=False))
print("reference logloss of coin flip x1000 =", round(1000 * np.log(2), 2))
