"""E05: availability with empirical return probabilities by consecutive games missed."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
import exp as E, context as C, players as PL
ROOT = os.path.join(os.path.dirname(__file__), "..")
g = pd.read_parquet(f"{ROOT}/data/processed/games.parquet")
cache = "experiments/.cache/tf_preturn.parquet"
if not os.path.exists(cache):
    pb = PL.load_player_games(g); pv = pd.read_parquet("experiments/.cache/pvals.parquet")
    PL.team_features(pb, g, pv).to_parquet(cache)
tf = pd.read_parquet(cache)
G = E.load_base().merge(pd.read_pickle('experiments/.cache/mkt_ratings.pkl'), on='game_id', how='left')
G = G[G.mkt_margin.notna()].reset_index(drop=True)
G = C.add_context(G, pd.read_parquet(f"{ROOT}/data/processed/games_all.parquet"))
for side, col in (("h", "home_id"), ("a", "away_id")):
    x = tf[["game_id", "team_id", "avail_off", "avail_def"]].rename(columns={"team_id": col, "avail_off": f"{side}_ao2", "avail_def": f"{side}_ad2"})
    G = G.merge(x, on=["game_id", col], how="left")
for c in ("h_ao2", "h_ad2", "a_ao2", "a_ad2"): G[c] = G[c].fillna(0.0)
G["avail_net2"] = (G.h_ao2 - G.h_ad2) - (G.a_ao2 - G.a_ad2)
G["avail_tot2"] = (G.h_ao2 + G.h_ad2) + (G.a_ao2 + G.a_ad2)
print("corr old/new avail_net:", np.corrcoef(G.avail_net, G.avail_net2)[0, 1].round(3), " sd old/new:", G.avail_net.std().round(3), G.avail_net2.std().round(3))
sides = ["rest_diff", "b2b_diff", "g7_diff", "trav_diff", "tz_a", "elev_home"]; tots = ["rest_sum", "b2b_any", "elev", "trav_a"]
M1, T1 = E.BASE_M + ["mkt_margin"] + sides, E.BASE_T + ["mkt_total"] + tots
sw = lambda cols, a, b: [b if c == a else c for c in cols]
rows = []
for lab, mc, tc in [("E04 base (old avail)", M1, T1),
                    ("new avail (replace)", sw(M1, "avail_net", "avail_net2"), sw(T1, "avail_tot", "avail_tot2")),
                    ("old + new avail", M1 + ["avail_net2"], T1 + ["avail_tot2"])]:
    r, _, _ = E.run(G, mc, tc, lab, show=False); rows.append(r)
print(pd.DataFrame(rows).set_index("exp").T.to_string())
