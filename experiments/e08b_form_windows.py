"""E08b: form window shape (rolling k vs exponentially weighted)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src")); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, statsmodels.api as sm
import exp as E
from e06_team_hca import G as G6, M1, T1
G = G6.merge(pd.read_pickle("experiments/.cache/conf_feats.pkl"), on="game_id", how="left")
M3, T3 = M1 + ["hca_team", "conf_m"], T1 + ["arena_tot", "conf_t"]
G["e_m"] = G.margin - G.eff_margin; G["e_t"] = G.total_pts - G.eff_total
long = pd.concat([pd.DataFrame({"game_id": G.game_id, "season": G.season, "date": G.date, "t": G.home_id, "em": G.e_m, "et": G.e_t}),
                  pd.DataFrame({"game_id": G.game_id, "season": G.season, "date": G.date, "t": G.away_id, "em": -G.e_m, "et": G.e_t})]).sort_values(["t", "date", "game_id"])
grp = long.groupby(["season", "t"])
specs = {}
for k in (4, 6, 10):
    specs[f"roll{k}"] = (grp.em.transform(lambda x: x.shift().rolling(k, min_periods=1).mean()), grp.et.transform(lambda x: x.shift().rolling(k, min_periods=1).mean()))
for hl in (3, 6):
    specs[f"ewm{hl}"] = (grp.em.transform(lambda x: x.shift().ewm(halflife=hl).mean()), grp.et.transform(lambda x: x.shift().ewm(halflife=hl).mean()))
# season-to-date mean (all previous games) - checks whether it's "recent" or just "rating bias"
specs["season"] = (grp.em.transform(lambda x: x.shift().expanding().mean()), grp.et.transform(lambda x: x.shift().expanding().mean()))
rows = []
r0, _, _ = E.run(G, M3, T3, "E07 base", show=False); rows.append(r0)
for name, (fm, ft) in specs.items():
    long["fm"], long["ft"] = fm.fillna(0).to_numpy(), ft.fillna(0).to_numpy()
    key = long.set_index(["game_id", "t"])
    for s, col in (("h", "home_id"), ("a", "away_id")):
        idx = pd.MultiIndex.from_arrays([G.game_id, G[col]]); G[f"{s}_fm"] = key.fm.reindex(idx).to_numpy(); G[f"{s}_ft"] = key.ft.reindex(idx).to_numpy()
    G[f"fm_{name}"] = G.h_fm - G.a_fm; G[f"ft_{name}"] = G.h_ft + G.a_ft
    r, _, _ = E.run(G, M3 + [f"fm_{name}"], T3 + [f"ft_{name}"], name, show=False); rows.append(r)
print(pd.DataFrame(rows).set_index("exp").T.drop(index=["roi_sp_2024", "roi_to_2024"]).to_string())
X = E.design(G, M3 + ["fm_roll6"]); f = G.season.isin(E.DEV)
print("coef fm_roll6:", sm.OLS(G.margin[f], X[f]).fit().params["fm_roll6"].round(3), " ft_roll6:", sm.OLS(G.total_pts[f], E.design(G, T3 + ["ft_roll6"])[f]).fit().params["ft_roll6"].round(3))
G[["game_id", "fm_roll6", "ft_roll6"]].to_pickle("experiments/.cache/form_feats.pkl")
