"""E01: market-derived team ratings from previous games' closing lines, as stage-2 features."""
import os, sys
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from multiprocessing import Pool
import exp as E
from ratings import CompParams, run_component

ROOT = os.path.join(os.path.dirname(__file__), "..")
g = pd.read_parquet(f"{ROOT}/data/processed/games.parquet")
tg = pd.read_parquet(f"{ROOT}/data/processed/teamgame.parquet")
n = len(g)
L, T = g.close_spread.to_numpy(float), g.close_total.to_numpy(float)
tg["mkt_pts"] = np.concatenate([(T - L) / 2, (T + L) / 2])
seasons = sorted(g.season.unique())

def mk(args):
    lam, tau, rho = args
    P = CompParams(lam_off=lam, lam_def=lam, tau=tau, rho_off=rho, rho_def=rho, lam_h=30)
    p, _ = run_component(tg, "mkt", P, seasons=seasons[1:], burn_in_season=seasons[0])
    return args, p

if __name__ == "__main__":
    grid = [(lam, tau, 0.7) for lam in (1, 3) for tau in (15, 45, 1000)]
    with Pool(4) as pool:
        res = pool.map(mk, grid)
    G = E.load_base()
    rows = []
    for (lam, tau, rho), p in res:
        m = pd.DataFrame({"game_id": g.game_id, "mkt_margin": p[:n] - p[n:], "mkt_total": p[:n] + p[n:]})
        H = G.drop(columns=[c for c in ("mkt_margin", "mkt_total") if c in G]).merge(m, on="game_id", how="left")
        H = H[H.mkt_margin.notna()].reset_index(drop=True)
        r, _, _ = E.run(H, E.BASE_M + ["mkt_margin"], E.BASE_T + ["mkt_total"], f"mkt lam={lam} tau={tau}", show=False)
        rows.append(r)
    base, _, _ = E.run(H, E.BASE_M, E.BASE_T, "baseline (same rows)", show=False)
    print(pd.DataFrame([base] + rows).set_index("exp").T.to_string())
    # cache the configuration used by later experiments (lam=1, no decay, rho=0.85)

    P = CompParams(lam_off=1, lam_def=1, tau=1000, rho_off=0.85, rho_def=0.85, lam_h=30)
    p, _ = run_component(tg, "mkt", P, seasons=seasons[1:], burn_in_season=seasons[0])
    os.makedirs("experiments/.cache", exist_ok=True)
    pd.DataFrame({"game_id": g.game_id, "mkt_margin": p[:n] - p[n:], "mkt_total": p[:n] + p[n:]}).to_pickle(
        "experiments/.cache/mkt_ratings.pkl")
