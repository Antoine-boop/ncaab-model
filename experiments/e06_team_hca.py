"""E06: team-specific home-court edge and arena scoring effects from the previous 3 seasons (shrunk)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
import exp as E, context as C
ROOT = os.path.join(os.path.dirname(__file__), "..")
P = pd.read_parquet(f"{ROOT}/data/processed/predictions.parquet")
P = P[P.margin.notna() & P.eff_margin.notna()].copy()
P["e_m"] = P.margin - P.eff_margin            # raw walk-forward residuals (no fitted stage-2 -> no leakage)
P["e_t"] = P.total_pts - P.eff_total
def team_dev(season, k_m=40, k_t=40):
    h = P[(P.season < season) & (P.season >= season - 3) & ~P.neutral]
    home = h.groupby("home_id").agg(sm=("e_m", "sum"), n=("e_m", "size"), st=("e_t", "sum"))
    away = h.groupby("away_id").agg(sm=("e_m", "sum"), n=("e_m", "size"))
    teams = home.index.union(away.index)
    hs, hn = home.sm.reindex(teams).fillna(0), home.n.reindex(teams).fillna(0)
    as_, an = (-away.sm).reindex(teams).fillna(0), away.n.reindex(teams).fillna(0)
    # home-vs-road split relative to the league HCA already in the model, shrunk
    hdev = 0.5 * (hs / (hn + k_m) - as_ / (an + k_m))
    tdev = home.st.reindex(teams).fillna(0) / (hn + k_t)       # arena scoring effect
    return hdev, tdev
G = E.load_base().merge(pd.read_pickle('experiments/.cache/mkt_ratings.pkl'), on='game_id', how='left')
G = G[G.mkt_margin.notna()].reset_index(drop=True)
G = C.add_context(G, pd.read_parquet(f"{ROOT}/data/processed/games_all.parquet"))
G["hca_team"] = 0.0; G["arena_tot"] = 0.0
for s in sorted(G.season.unique()):
    hdev, tdev = team_dev(s)
    m = (G.season == s) & ~G.neutral
    G.loc[m, "hca_team"] = G.loc[m, "home_id"].map(hdev).fillna(0).to_numpy() + G.loc[m, "away_id"].map(hdev).fillna(0).to_numpy()
    G.loc[m, "arena_tot"] = G.loc[m, "home_id"].map(tdev).fillna(0).to_numpy()
print(G[["hca_team", "arena_tot"]].describe().round(2).T[["std", "min", "max"]])
sides = ["rest_diff", "b2b_diff", "g7_diff", "trav_diff", "tz_a", "elev_home"]; tots = ["rest_sum", "b2b_any", "elev", "trav_a"]
M1, T1 = E.BASE_M + ["mkt_margin"] + sides, E.BASE_T + ["mkt_total"] + tots
if __name__ == "__main__":
  rows = []
  for lab, mc, tc in [("E04 base", M1, T1), ("+team HCA", M1 + ["hca_team"], T1), ("+arena total", M1, T1 + ["arena_tot"]),
                      ("+both", M1 + ["hca_team"], T1 + ["arena_tot"])]:
    r, _, _ = E.run(G, mc, tc, lab, show=False); rows.append(r)
  print(pd.DataFrame(rows).set_index("exp").T.to_string())
