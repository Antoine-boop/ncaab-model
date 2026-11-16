"""E02: mismatch nonlinearity in stage 2 (signed square, interactions with |margin|)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
import exp as E
G = E.load_base()
m = pd.read_pickle('experiments/.cache/mkt_ratings.pkl')
G = G.merge(m, on='game_id', how='left'); G = G[G.mkt_margin.notna()].reset_index(drop=True)
G["sq_eff"] = G.eff_margin * G.eff_margin.abs() / 10
G["cube_eff"] = G.eff_margin ** 3 / 100
G["abs_eff"] = G.eff_margin.abs()
G["abs_eff_x_tempo"] = G.abs_eff * G.tempo_hat / 68
G["blowout"] = np.maximum(G.abs_eff - 15, 0)
M0, T0 = E.BASE_M + ["mkt_margin"], E.BASE_T + ["mkt_total"]
rows = []
for lab, mc, tc in [("E01 base", M0, T0),
                    ("+sq_eff (margin)", M0 + ["sq_eff"], T0),
                    ("+cube_eff (margin)", M0 + ["cube_eff"], T0),
                    ("+abs_eff (total)", M0, T0 + ["abs_eff"]),
                    ("+blowout (total)", M0, T0 + ["blowout"]),
                    ("+sq + abs", M0 + ["sq_eff"], T0 + ["abs_eff", "blowout"])]:
    r, pm, pt = E.run(G, mc, tc, lab, show=False); rows.append(r)
print(pd.DataFrame(rows).set_index("exp").T.to_string())
