"""E04: rest, schedule density, travel, time zones, altitude."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd, statsmodels.api as sm
import exp as E, context as C
ROOT = os.path.join(os.path.dirname(__file__), "..")
G = E.load_base().merge(pd.read_pickle('experiments/.cache/mkt_ratings.pkl'), on='game_id', how='left')
G = G[G.mkt_margin.notna()].reset_index(drop=True)
G = C.add_context(G, pd.read_parquet(f"{ROOT}/data/processed/games_all.parquet"))
print(G[["rest_h", "rest_a", "g7_h", "trav_h", "trav_a", "tz_a", "elev", "elev_home"]].describe().round(2).T[["mean", "std", "max"]].to_string())
M0, T0 = E.BASE_M + ["mkt_margin"], E.BASE_T + ["mkt_total"]
sides = ["rest_diff", "b2b_diff", "g7_diff", "trav_diff", "tz_a", "elev_home"]
tots = ["rest_sum", "b2b_any", "elev", "trav_a"]
rows = []
for lab, mc, tc in [("E01 base", M0, T0), ("+context sides", M0 + sides, T0), ("+context totals", M0, T0 + tots),
                    ("+both", M0 + sides, T0 + tots)]:
    r, _, _ = E.run(G, mc, tc, lab, show=False); rows.append(r)
print(pd.DataFrame(rows).set_index("exp").T.to_string())
X = E.design(G, M0 + sides); f = G.season.isin(E.DEV)
fit = sm.OLS(G.margin[f], X[f]).fit(); print("margin coefs:", fit.params[sides].round(3).to_dict(), "\n t:", fit.tvalues[sides].round(1).to_dict())
X = E.design(G, T0 + tots); fit = sm.OLS(G.total_pts[f], X[f]).fit(); print("total coefs:", fit.params[tots].round(3).to_dict(), "\n t:", fit.tvalues[tots].round(1).to_dict())
