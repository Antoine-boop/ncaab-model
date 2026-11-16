"""E07: conference-level residual signals, walk-forward within season (raw model residuals, shrunk)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
import exp as E, context as C
sys.path.insert(0, os.path.dirname(__file__))
from e06_team_hca import G as G6, M1, T1   # reuse E06 frame (+hca_team, arena_tot)
G = G6.copy()
G["e_m"] = G.margin - G.eff_margin; G["e_t"] = G.total_pts - G.eff_total
G["conf_m"] = 0.0; G["conf_t"] = 0.0
K_M, K_T = 30.0, 30.0
for s, d in G.groupby("season"):
    d = d.sort_values("date")
    nonconf = d[~d.conf_game & (d.home_conf != d.away_conf)]
    conf = d[d.conf_game]
    for D in d.date.unique():
        prev_nc = nonconf[nonconf.date < D]
        # conference residual from each conference's perspective in non-conference games
        r = pd.concat([pd.DataFrame({"c": prev_nc.home_conf, "e": prev_nc.e_m}), pd.DataFrame({"c": prev_nc.away_conf, "e": -prev_nc.e_m})])
        agg = r.groupby("c").e.agg(["sum", "size"]); cm = agg["sum"] / (agg["size"] + K_M)
        prev_c = conf[conf.date < D]
        at = prev_c.groupby("home_conf").e_t.agg(["sum", "size"]); ct = at["sum"] / (at["size"] + K_T)
        idx = d.index[d.date == D]
        G.loc[idx, "conf_m"] = (G.loc[idx, "home_conf"].map(cm).fillna(0) - G.loc[idx, "away_conf"].map(cm).fillna(0)).to_numpy()
        cg = G.loc[idx, "conf_game"].to_numpy()
        G.loc[idx, "conf_t"] = np.where(cg, G.loc[idx, "home_conf"].map(ct).fillna(0).to_numpy(), 0.0)
print(G[["conf_m", "conf_t"]].describe().round(2).T[["std", "min", "max"]])
M2, T2 = M1 + ["hca_team"], T1 + ["arena_tot"]
rows = []
for lab, mc, tc in [("E06 base", M2, T2), ("+conf_m", M2 + ["conf_m"], T2), ("+conf_t", M2, T2 + ["conf_t"]), ("+both", M2 + ["conf_m"], T2 + ["conf_t"])]:
    r, _, _ = E.run(G, mc, tc, lab, show=False); rows.append(r)
print(pd.DataFrame(rows).set_index("exp").T.to_string())
G[["game_id", "conf_m", "conf_t"]].to_pickle("experiments/.cache/conf_feats.pkl")
