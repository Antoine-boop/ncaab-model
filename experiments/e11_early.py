"""E11: early-season interactions for market rating, form, HCA."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd, statsmodels.api as sm
import exp as E
from common import build, M_KEEP, T_KEEP
G = build()
for c in ("mkt_margin", "form_m", "mkt_total", "form_t", "conf_m", "avail_net"):
    G[f"early_{c}"] = G.early * G[c]
G["nov"] = (G.date.dt.month == 11).astype(float)
for c in ("mkt_margin", "eff_margin", "mkt_total", "comp_total"):
    G[f"nov_{c}"] = G.nov * G[c]
rows = []
for lab, mc, tc in [("E08 keep", M_KEEP, T_KEEP),
                    ("+early x mkt", M_KEEP + ["early_mkt_margin"], T_KEEP + ["early_mkt_total"]),
                    ("+early x form", M_KEEP + ["early_form_m"], T_KEEP + ["early_form_t"]),
                    ("+early x (mkt, form, conf, avail)", M_KEEP + ["early_mkt_margin", "early_form_m", "early_conf_m", "early_avail_net"], T_KEEP + ["early_mkt_total", "early_form_t"]),
                    ("+nov x (mkt, eff)", M_KEEP + ["nov_mkt_margin", "nov_eff_margin"], T_KEEP + ["nov_mkt_total", "nov_comp_total"])]:
    r, _, _ = E.run(G, mc, tc, lab, show=False); rows.append(r)
print(pd.DataFrame(rows).set_index("exp").T.to_string())
X = E.design(G, M_KEEP + ["early_mkt_margin"]); f = G.season.isin(E.DEV)
print(sm.OLS(G.margin[f], X[f]).fit().params[["mkt_margin", "early_mkt_margin", "eff_margin", "early_eff_margin"]].round(3).to_dict())
