"""E03: multiplicative (log-PPP) efficiency ratings vs additive PPP."""
import os, sys
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from multiprocessing import Pool
import exp as E, model as MD
from ratings import run_component
ROOT = os.path.join(os.path.dirname(__file__), "..")
g = pd.read_parquet(f"{ROOT}/data/processed/games.parquet"); tg = pd.read_parquet(f"{ROOT}/data/processed/teamgame.parquet"); n = len(g)
tg["lppp_num"] = np.log(np.clip(tg.pts / tg.poss, 0.2, 3.0)) * tg.poss
seasons = sorted(g.season.unique()); prm = MD.load_params()
def go(comp):
    P = MD.comp_params("ppp", prm["ppp"])
    p, _ = run_component(tg, comp, P, seasons=seasons[1:], burn_in_season=seasons[0])
    return comp, p
if __name__ == "__main__":
    with Pool(2) as pool:
        res = dict(pool.map(go, ["ppp", "lppp"]))
    G = E.load_base().merge(pd.read_pickle('experiments/.cache/mkt_ratings.pkl'), on='game_id', how='left')
    G = G[G.mkt_margin.notna()].reset_index(drop=True)
    tempo = G.tempo_hat.to_numpy()
    for comp, p in res.items():
        d = pd.DataFrame({"game_id": g.game_id, f"h_{comp}": p[:n], f"a_{comp}": p[n:]})
        G = G.merge(d, on="game_id", how="left")
    G["eff_m_add"] = G.tempo_hat * (G.h_ppp - G.a_ppp); G["eff_t_add"] = G.tempo_hat * (G.h_ppp + G.a_ppp)
    G["eff_m_mul"] = G.tempo_hat * (np.exp(G.h_lppp) - np.exp(G.a_lppp)); G["eff_t_mul"] = G.tempo_hat * (np.exp(G.h_lppp) + np.exp(G.a_lppp))
    M0, T0 = E.BASE_M + ["mkt_margin"], E.BASE_T + ["mkt_total"]
    swap = lambda cols, a, b: [b if c == a else c for c in cols]
    rows = []
    for lab, mc, tc in [("E01 base (pipeline eff, roster priors)", M0, T0),
                        ("eff=additive (team priors)", swap(M0, "eff_margin", "eff_m_add"), swap(T0, "eff_total", "eff_t_add")),
                        ("eff=multiplicative", swap(M0, "eff_margin", "eff_m_mul"), swap(T0, "eff_total", "eff_t_mul")),
                        ("base + multiplicative", M0 + ["eff_m_mul"], T0 + ["eff_t_mul"])]:
        r, _, _ = E.run(G, mc, tc, lab, show=False); rows.append(r)
    print(pd.DataFrame(rows).set_index("exp").T.to_string())
