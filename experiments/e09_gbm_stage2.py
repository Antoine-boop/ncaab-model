"""E09: LightGBM on the residual of the linear stage 2 (same OOS folds)."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd, lightgbm as lgb
import exp as E
from common import build, M_KEEP, T_KEEP
G = build()
pm, pt = E.stage2_oos(G, M_KEEP, T_KEEP)
r0 = dict(exp="linear (E08)") | E.metrics(G, pm, pt)
extra = ["early", "min_games", "h_n_games", "a_n_games", "neutral_f", "conf_f", "post", "rest_h", "rest_a", "elev", "tempo_hat"]
G["neutral_f"] = G.neutral.astype(float); G["conf_f"] = G.conf_game.astype(float); G["post"] = (G.season_type == 3).astype(float)
G["month"] = G.date.dt.month
feats_m = [c for c in dict.fromkeys(M_KEEP + extra + ["month", "Mo_spread"]) if c not in ("Mo_spread",)]
feats_t = [c for c in dict.fromkeys(T_KEEP + extra + ["month"])]
Xm, Xt = E.design(G, feats_m).drop(columns="const"), E.design(G, feats_t).drop(columns="const")
seas = G.season.to_numpy()
def oos_gbm(X, y, base, params, rounds):
    out = base.copy()
    for s in sorted(np.unique(seas)):
        fit = [x for x in E.DEV if x != s] if s <= 2022 else list(range(2016, s))
        f = np.isin(seas, fit) & np.isfinite(base); p = (seas == s) & np.isfinite(base)
        mdl = lgb.train(params, lgb.Dataset(X[f], y[f] - base[f]), num_boost_round=rounds)
        out[p] = base[p] + mdl.predict(X[p])
    return out
rows = [r0]
for lr, rounds, leaves in ((0.03, 150, 15), (0.03, 400, 15), (0.02, 300, 31)):
    params = dict(objective="huber", alpha=12.0, learning_rate=lr, num_leaves=leaves, min_data_in_leaf=300, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=4)
    pm2 = oos_gbm(Xm.to_numpy(float), G.margin.to_numpy(float), pm, params, rounds)
    params_t = dict(params, alpha=18.0)
    pt2 = oos_gbm(Xt.to_numpy(float), G.total_pts.to_numpy(float), pt, params_t, rounds)
    rows.append(dict(exp=f"gbm lr={lr} n={rounds} leaves={leaves}") | E.metrics(G, pm2, pt2))
print(pd.DataFrame(rows).set_index("exp").T.to_string())
