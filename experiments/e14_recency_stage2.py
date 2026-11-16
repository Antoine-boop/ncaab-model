"""E14: recency-weighted stage-2 fits (weight 'decay^(years back)')."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
import exp as E
from common import build, M_KEEP, T_KEEP
G = build()
def stage2_w(G, mcols, tcols, decay):
    pm, pt = np.full(len(G), np.nan), np.full(len(G), np.nan)
    Xm, Xt = E.design(G, mcols).to_numpy(float), E.design(G, tcols).to_numpy(float)
    ym, yt, seas = G.margin.to_numpy(float), G.total_pts.to_numpy(float), G.season.to_numpy()
    ok = np.isfinite(Xm).all(1) & np.isfinite(Xt).all(1)
    for s in sorted(np.unique(seas)):
        fit = [x for x in E.DEV if x != s] if s <= 2022 else list(range(2016, s))
        f = np.isin(seas, fit) & ok; p = (seas == s) & ok
        w = np.sqrt(decay ** np.abs(s - seas[f]))
        pm[p] = Xm[p] @ np.linalg.lstsq(Xm[f] * w[:, None], ym[f] * w, rcond=None)[0]
        pt[p] = Xt[p] @ np.linalg.lstsq(Xt[f] * w[:, None], yt[f] * w, rcond=None)[0]
    return pm, pt
rows = []
for dec in (1.0, 0.85, 0.7):
    pm, pt = stage2_w(G, M_KEEP, T_KEEP, dec)
    rows.append(dict(exp=f"decay {dec}") | E.metrics(G, pm, pt))
print(pd.DataFrame(rows).set_index("exp").T.to_string())
