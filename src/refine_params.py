"""
Refine carry-over (rho) around each component's best grid point (rho hit the grid edge), then write
reports/best_params.json used by model.py. DEV seasons only.
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import multiprocessing as mp

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tune_components as T  # noqa: E402

ROOT = T.ROOT

if __name__ == "__main__":
    jobs = []
    best = {}
    for comp in T.PER_GAME_W:
        f = os.path.join(ROOT, "reports", f"tune_{comp}.csv")
        b = pd.read_csv(f).sort_values("loss").iloc[0]
        best[comp] = b
        rhos = (0.85, 0.92, 1.0) if comp == "ppp" else (0.6, 0.75, 0.9)
        lams = (b.lam_off_g, b.lam_off_g * 1.5) if comp == "ppp" else (b.lam_off_g,)
        for r in rhos:
            for lo in lams:
                ld = lo if comp == "ppp" else b.lam_def_g
                jobs.append((comp, lo, ld, b.tau, r, r))
    with mp.Pool(4, initializer=T._init) as pool:
        res = pd.DataFrame(pool.map(T._eval, jobs))
    res.to_csv(os.path.join(ROOT, "reports", "refine_rho.csv"), index=False)
    out = {}
    for comp, b in best.items():
        r = res[res.comp == comp].sort_values("loss").iloc[0]
        pick = r if r.loss < b.loss else b
        out[comp] = dict(lam_off_g=float(pick.lam_off_g), lam_def_g=float(pick.lam_def_g), tau=float(pick.tau),
                         rho_off=float(pick.rho_off), rho_def=float(pick.rho_def))
        print(comp, out[comp], "loss %.6g (grid best %.6g)" % (min(r.loss, b.loss), b.loss), flush=True)
    with open(os.path.join(ROOT, "reports", "best_params.json"), "w") as fh:
        json.dump(out, fh, indent=1)
