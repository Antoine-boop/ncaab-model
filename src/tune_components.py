"""
Grid-search ridge hyper-parameters per component on DEV seasons only (walk-forward predictive loss).

    python src/tune_components.py ppp tempo tov ...      -> writes reports/tune_<comp>.csv
"""
from __future__ import annotations

import itertools
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import multiprocessing as mp
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ratings import CompParams, component_arrays, run_component  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = list(range(2016, 2023))
BURN = 2015

# typical weight per team-game, used to express prior strength in "games"
PER_GAME_W = {"ppp": 70, "tempo": 1, "tov": 70, "orb": 30, "ftr": 58, "ftp": 20, "r3": 58, "p3": 22, "p2": 35}

_TG = None


def _init():
    global _TG
    _TG = pd.read_parquet(os.path.join(ROOT, "data", "processed", "teamgame.parquet"))


def _eval(args):
    comp, lo, ld, tau, ro, rd = args
    tg = _TG
    pw = PER_GAME_W[comp]
    lam_h = 30 * pw
    P = CompParams(lam_off=lo * pw, lam_def=ld * pw, tau=tau, rho_off=ro, rho_def=rd, lam_h=lam_h)
    pred, _ = run_component(tg, comp, P, seasons=DEV, burn_in_season=BURN)
    y, w, _ = component_arrays(tg, comp)
    m = np.isfinite(pred) & tg.season.isin(DEV).to_numpy() & (w > 0)
    err = (y[m] - pred[m]) ** 2
    loss = float((w[m] * err).sum() / w[m].sum())
    # early-season (Nov) vs rest
    mon = tg.date.dt.month.to_numpy()[m]
    early = mon == 11
    le = float((w[m][early] * err[early]).sum() / w[m][early].sum())
    ll = float((w[m][~early] * err[~early]).sum() / w[m][~early].sum())
    return dict(comp=comp, lam_off_g=lo, lam_def_g=ld, tau=tau, rho_off=ro, rho_def=rd, loss=loss,
                loss_nov=le, loss_rest=ll)


def grid(comp):
    if comp == "tempo":
        lams = [(l, l) for l in (1, 2, 4, 8, 16)]
        rhos = [(r, r) for r in (0.5, 0.65, 0.8, 0.9)]
    elif comp == "ppp":
        lams = [(l, l) for l in (2, 4, 8, 16, 32)]
        rhos = [(r, r) for r in (0.4, 0.55, 0.7, 0.85)]
    else:
        lams = list(itertools.product((3, 10, 30, 100), (3, 10, 30, 100, 300)))
        rhos = [(r, r) for r in (0.3, 0.6)]
    taus = (30, 60, 120, 1000)
    return [(comp, lo, ld, t, ro, rd) for (lo, ld) in lams for t in taus for (ro, rd) in rhos]


if __name__ == "__main__":
    comps = sys.argv[1:] or list(PER_GAME_W)
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    with mp.Pool(4, initializer=_init) as pool:
        for comp in comps:
            res = pd.DataFrame(pool.map(_eval, grid(comp))).sort_values("loss")
            res.to_csv(os.path.join(ROOT, "reports", f"tune_{comp}.csv"), index=False)
            print(comp, "best:\n", res.head(5).to_string(index=False), flush=True)
