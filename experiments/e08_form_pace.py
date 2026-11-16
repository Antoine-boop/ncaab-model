"""E08: recent form (last-k residuals) and pace interactions for totals."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src")); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import exp as E
from e06_team_hca import G as G6, M1, T1
G = G6.merge(pd.read_pickle("experiments/.cache/conf_feats.pkl"), on="game_id", how="left")
M3, T3 = M1 + ["hca_team", "conf_m"], T1 + ["arena_tot", "conf_t"]
# --- form: mean raw residual over each team's previous k games this season (team perspective) ---
G["e_m"] = G.margin - G.eff_margin; G["e_t"] = G.total_pts - G.eff_total
long = pd.concat([pd.DataFrame({"game_id": G.game_id, "season": G.season, "date": G.date, "t": G.home_id, "em": G.e_m, "et": G.e_t}),
                  pd.DataFrame({"game_id": G.game_id, "season": G.season, "date": G.date, "t": G.away_id, "em": -G.e_m, "et": G.e_t})])
long = long.sort_values(["t", "date", "game_id"])
for k in (3, 6):
    long[f"fm{k}"] = long.groupby(["season", "t"]).em.transform(lambda x: x.shift().rolling(k, min_periods=1).mean()).fillna(0)
    long[f"ft{k}"] = long.groupby(["season", "t"]).et.transform(lambda x: x.shift().rolling(k, min_periods=1).mean()).fillna(0)
key = long.set_index(["game_id", "t"])
for k in (3, 6):
    for s, col in (("h", "home_id"), ("a", "away_id")):
        idx = pd.MultiIndex.from_arrays([G.game_id, G[col]])
        G[f"{s}_fm{k}"] = key[f"fm{k}"].reindex(idx).to_numpy(); G[f"{s}_ft{k}"] = key[f"ft{k}"].reindex(idx).to_numpy()
    G[f"form_m{k}"] = G[f"h_fm{k}"] - G[f"a_fm{k}"]; G[f"form_t{k}"] = G[f"h_ft{k}"] + G[f"a_ft{k}"]
# --- pace interactions ---
th, ta = G.h_ppp_eff * 0 + G.tempo_hat, None
G["tempo_absdiff"] = 0.0
tg_h = G.tempo_hat  # combined; we need team tempos -> approximate via components not available; use r3/ftr style proxies
G["pace_x_eff"] = (G.tempo_hat - 68) * (G.h_ppp_eff + G.a_ppp_eff - 2.06)
G["pace_sq"] = (G.tempo_hat - 68) ** 2
rows = []
for lab, mc, tc in [("E07 base", M3, T3), ("+form3", M3 + ["form_m3"], T3 + ["form_t3"]), ("+form6", M3 + ["form_m6"], T3 + ["form_t6"]),
                    ("+pace interactions", M3, T3 + ["pace_x_eff", "pace_sq"])]:
    r, _, _ = E.run(G, mc, tc, lab, show=False); rows.append(r)
print(pd.DataFrame(rows).set_index("exp").T.to_string())
