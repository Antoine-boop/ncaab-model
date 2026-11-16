"""
End-to-end bottom-up model.

  players  -> player values -> roster priors + availability adjustments
  box      -> per-component opponent-adjusted ratings (walk-forward ridge)
  ratings  -> component projections -> PPP via possession identity -> points (x tempo)
  stage-2  -> linear blend (fit on DEV seasons only) -> margin, total, P(home win)

Run:  python src/model.py          (writes data/processed/predictions.parquet)
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import players as PL  # noqa: E402
import priors as PR  # noqa: E402
from build_dataset import team_game_rows  # noqa: E402
from ratings import COMPONENTS, CompParams, ppp_from_components, run_component  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
DEV = list(range(2016, 2023))          # stage-2 + calibration fit seasons
BURN = 2015

PER_GAME_W = {"ppp": 70, "tempo": 1, "tov": 70, "orb": 30, "ftr": 58, "ftp": 20, "r3": 58, "p3": 22, "p2": 35}

# Defaults; overwritten by reports/best_params.json when present (written from tuning results)
DEFAULT_PARAMS = {
    "ppp": dict(lam_off_g=8, lam_def_g=8, tau=1000, rho_off=0.85, rho_def=0.85),
    "tempo": dict(lam_off_g=4, lam_def_g=4, tau=30, rho_off=0.8, rho_def=0.8),
    "tov": dict(lam_off_g=10, lam_def_g=10, tau=120, rho_off=0.6, rho_def=0.6),
    "orb": dict(lam_off_g=10, lam_def_g=10, tau=120, rho_off=0.6, rho_def=0.6),
    "ftr": dict(lam_off_g=10, lam_def_g=10, tau=120, rho_off=0.6, rho_def=0.6),
    "ftp": dict(lam_off_g=10, lam_def_g=300, tau=1000, rho_off=0.6, rho_def=0.3),
    "r3": dict(lam_off_g=10, lam_def_g=10, tau=120, rho_off=0.6, rho_def=0.6),
    "p3": dict(lam_off_g=30, lam_def_g=100, tau=1000, rho_off=0.6, rho_def=0.3),
    "p2": dict(lam_off_g=10, lam_def_g=10, tau=120, rho_off=0.6, rho_def=0.6),
}


def load_params():
    p = {k: dict(v) for k, v in DEFAULT_PARAMS.items()}
    f = os.path.join(ROOT, "reports", "best_params.json")
    if os.path.exists(f):
        with open(f) as fh:
            p.update(json.load(fh))
    return p


def comp_params(comp, d):
    pw = PER_GAME_W[comp]
    return CompParams(lam_off=d["lam_off_g"] * pw, lam_def=d["lam_def_g"] * pw, tau=d["tau"],
                      rho_off=d["rho_off"], rho_def=d["rho_def"], lam_h=30 * pw)


def upcoming_rows(up: pd.DataFrame) -> pd.DataFrame:
    """Team-game rows for unplayed games (zero weight, still predicted)."""
    u = up.copy()
    for s in ("h", "a"):
        for c in ("fga", "fgm", "tpa", "tpm", "fta", "ftm", "oreb", "dreb", "tov"):
            u[f"{s}_{c}"] = 0.0
    for c in ("poss", "tempo", "minutes", "home_score", "away_score"):
        u[c] = 0.0
    tg = team_game_rows(u)
    tg["tempo"] = np.nan
    return tg


def run_pipeline(games: pd.DataFrame, upcoming: pd.DataFrame | None = None, params=None, verbose=True):
    params = params or load_params()
    seasons = sorted(games.season.unique())
    allg = games if upcoming is None or upcoming.empty else pd.concat([games, upcoming], ignore_index=True)
    allg = allg.sort_values(["date", "game_id"]).reset_index(drop=True)
    tg = team_game_rows(allg)
    if upcoming is not None and not upcoming.empty:
        is_up = allg.game_id.isin(upcoming.game_id).to_numpy()
        n = len(allg)
        mask = np.concatenate([is_up, is_up])
        for c in ("poss", "pts", "fga", "fgm", "tpa", "tpm", "fta", "ftm", "oreb", "tov", "opp_dreb", "twa", "twm", "miss"):
            tg.loc[mask, c] = 0.0
        tg.loc[mask, "tempo"] = np.nan
    n = len(allg)
    seas = seasons[1:]
    burn = seasons[0]

    # ---- pass 1: efficiency with team-level priors -> end-of-season ratings -> player values ----
    pp = comp_params("ppp", params["ppp"])
    _, fin1 = run_component(tg, "ppp", pp, seasons=seas, burn_in_season=burn)
    team_final = {s: pd.DataFrame({"team_id": f["teams"], "o": 100 * f["off"], "d": 100 * f["de"]})
                  for s, f in fin1.items()}
    pb = PL.load_player_games(games)
    pv = PL.player_season_values(pb, team_final)
    tf = PL.team_features(pb, games, pv, upcoming=upcoming)
    if verbose:
        print("player features", tf.shape, flush=True)

    # ---- pass 2: efficiency with bottom-up roster priors ----
    dev_seasons = [s for s in DEV if s in seasons]
    pm, tm = PR.fit_prior_models(tf, allg, fin1, dev_seasons)
    prior_fn = PR.make_prior_fn(tf, allg, pm, tm)
    preds = {}
    preds["ppp"], fin_ppp = run_component(tg, "ppp", pp, seasons=seas, burn_in_season=burn, prior_fn=prior_fn,
                                          return_state=False)
    finals = {"ppp": fin_ppp}
    for comp in COMPONENTS:
        if comp == "ppp":
            continue
        preds[comp], finals[comp] = run_component(tg, comp, comp_params(comp, params[comp]), seasons=seas,
                                                  burn_in_season=burn)
        if verbose:
            print("component", comp, "done", flush=True)

    # ---- assemble game-level features ----
    G = allg.copy()
    G["tempo_hat"] = preds["tempo"][:n]
    G["h_ppp_eff"], G["a_ppp_eff"] = preds["ppp"][:n], preds["ppp"][n:]
    comps = {}
    for side, sl in (("h", slice(0, n)), ("a", slice(n, 2 * n))):
        c = {k: np.clip(preds[k][sl], 1e-3, 0.999) for k in ("tov", "orb", "ftr", "ftp", "r3", "p3", "p2")}
        comps[side] = c
        G[f"{side}_ppp_comp"] = ppp_from_components(**c)
        for k, v in c.items():
            G[f"{side}_{k}_hat"] = v
    G["eff_margin"] = G.tempo_hat * (G.h_ppp_eff - G.a_ppp_eff)
    G["eff_total"] = G.tempo_hat * (G.h_ppp_eff + G.a_ppp_eff)
    G["comp_margin"] = G.tempo_hat * (G.h_ppp_comp - G.a_ppp_comp)
    G["comp_total"] = G.tempo_hat * (G.h_ppp_comp + G.a_ppp_comp)

    feat_cols = ["avail_off", "avail_def", "missing_last", "roster_off", "roster_def", "seen_new_share", "ret_share",
                 "n_games"]
    for side, col in (("h", "home_id"), ("a", "away_id")):
        x = tf[["game_id", "team_id"] + [c for c in feat_cols if c in tf.columns]].rename(columns={"team_id": col})
        x = x.rename(columns={c: f"{side}_{c}" for c in feat_cols})
        G = G.merge(x, on=["game_id", col], how="left")
    for c in ("avail_off", "avail_def", "missing_last"):
        G[f"h_{c}"] = G[f"h_{c}"].fillna(0.0)
        G[f"a_{c}"] = G[f"a_{c}"].fillna(0.0)
    G["avail_net"] = (G.h_avail_off - G.h_avail_def) - (G.a_avail_off - G.a_avail_def)
    G["avail_tot"] = (G.h_avail_off + G.h_avail_def) + (G.a_avail_off + G.a_avail_def)
    G["min_games"] = np.minimum(G.h_n_games.fillna(0), G.a_n_games.fillna(0))
    G["early"] = np.exp(-G.min_games / 6.0)       # ~1 for first games, decays with games played
    G["loc_h"] = np.where(G.neutral, 0, 1)
    return G, dict(tf=tf, pv=pv, finals=finals, prior_models=pm)


# --------------------------------------------------------------------------------------------
# stage 2
# --------------------------------------------------------------------------------------------
MARGIN_X = ["eff_margin", "comp_margin", "avail_net", "early_eff_margin", "early_comp_margin",
            "m_tov", "m_orb", "m_ftr", "m_ftp", "m_r3", "m_p3", "m_p2"]
TOTAL_X = ["eff_total", "comp_total", "avail_tot", "early_eff_total", "early_comp_total",
           "ot_exp", "neutral_f", "post", "conf_f", "tempo_hat",
           "t_tov", "t_orb", "t_ftr", "t_ftp", "t_r3", "t_p3", "t_p2"]


def _design(G, cols):
    X = G.copy()
    for base in ("eff_margin", "comp_margin", "eff_total", "comp_total"):
        X[f"early_{base}"] = X.early * X[base]
    # expected overtime points: P(OT) falls with projected |margin| (logit fit on dev seasons), ~21 pts per OT game
    m0 = X["eff_margin"].abs()
    X["ot_exp"] = 21.0 * (X.tempo_hat / 68.0) / (1 + np.exp(2.4007 + 0.014 * m0 + 0.0033 * m0 ** 2))
    X["neutral_f"] = X.neutral.astype(float)
    X["post"] = (X.season_type == 3).astype(float)
    X["conf_f"] = X.conf_game.astype(float)
    # pace-scaled component sums: lets stage 2 re-weight the bottom-up pieces of the total
    for c in ("tov", "orb", "ftr", "ftp", "r3", "p3", "p2"):
        X[f"t_{c}"] = X.tempo_hat * (X[f"h_{c}_hat"] + X[f"a_{c}_hat"])
        X[f"m_{c}"] = X.tempo_hat * (X[f"h_{c}_hat"] - X[f"a_{c}_hat"])
    return sm.add_constant(X[cols], has_constant="add")


def fit_stage2(G, fit_seasons=DEV):
    tr = G[G.season.isin(fit_seasons) & G.margin.notna() & G.eff_margin.notna() & G.comp_margin.notna()]
    mm = sm.OLS(tr.margin, _design(tr, MARGIN_X)).fit()
    # totals: residual excludes OT inflation by modelling regulation-equivalent expectation + OT term
    mt = sm.OLS(tr.total_pts, _design(tr, TOTAL_X)).fit()
    resid = tr.margin - mm.predict(_design(tr, MARGIN_X))
    sig_m = float(np.sqrt(np.mean(resid ** 2)))
    rt = tr.total_pts - mt.predict(_design(tr, TOTAL_X))
    sig_t = float(np.sqrt(np.mean(rt ** 2)))
    return dict(margin=mm, total=mt, sig_m=sig_m, sig_t=sig_t)


def apply_stage2(G, s2):
    G = G.copy()
    G["pred_margin"] = s2["margin"].predict(_design(G, MARGIN_X))
    G["pred_total"] = s2["total"].predict(_design(G, TOTAL_X))
    G["p_home"] = norm.cdf(G.pred_margin / s2["sig_m"])
    return G


def main():
    games = pd.read_parquet(os.path.join(PROC, "games.parquet"))
    G, aux = run_pipeline(games)
    s2 = fit_stage2(G)
    print(s2["margin"].params.round(4).to_dict())
    print(s2["total"].params.round(4).to_dict())
    print("sigma margin %.2f total %.2f" % (s2["sig_m"], s2["sig_t"]))
    G = apply_stage2(G, s2)
    G.to_parquet(os.path.join(PROC, "predictions.parquet"), index=False)
    pd.to_pickle(s2, os.path.join(PROC, "stage2.pkl"))
    return G


if __name__ == "__main__":
    main()
