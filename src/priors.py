"""
Preseason / early-season team priors built bottom-up from players.

prior_o = a + b1*o_last + b2*o_last*ret_share + b3*roster_off + b4*new_share + b5*roster_off*new_share
prior_d = (same with defense)
Teams with no games yet this season fall back to a team-only regression (o_last, d_last).
Coefficients are fit on DEV seasons only; targets are end-of-season ridge ratings (points / 100 poss).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

FORM = {
    "o": "o ~ o_last + o_last:ret_share + roster_off + seen_new_share + roster_off:seen_new_share",
    "d": "d ~ d_last + d_last:ret_share + roster_def + seen_new_share + roster_def:seen_new_share",
}
FORM_TEAM = {"o": "o ~ o_last", "d": "d ~ d_last"}


def final_to_frame(fin: dict) -> pd.DataFrame:
    return pd.concat([pd.DataFrame({"team_id": f["teams"], "o": 100 * f["off"], "d": 100 * f["de"]}).assign(season=s)
                      for s, f in fin.items()], ignore_index=True)


def fit_prior_models(tf: pd.DataFrame, games: pd.DataFrame, fin: dict, dev_seasons, k_fit=3):
    ff = final_to_frame(fin)
    last = ff.assign(season=ff.season + 1).rename(columns={"o": "o_last", "d": "d_last"})
    x = tf.merge(games[["game_id", "date"]], on="game_id").sort_values(["season", "team_id", "date"])
    x["k"] = x.groupby(["season", "team_id"]).cumcount()
    x = x[x.k == k_fit].merge(ff, on=["season", "team_id"]).merge(last, on=["season", "team_id"])
    x = x[x.season.isin(dev_seasons)].dropna(subset=["roster_off", "roster_def"])
    models = {c: smf.ols(FORM[c], x).fit() for c in ("o", "d")}
    team_models = {c: smf.ols(FORM_TEAM[c], x).fit() for c in ("o", "d")}
    return models, team_models


def make_prior_fn(tf: pd.DataFrame, games: pd.DataFrame, models, team_models, new_team_pct=0.2):
    """Returns prior_fn(season, teams, prev) -> callable(date) -> (prior_off, prior_def) in PPP units."""
    x = tf.merge(games[["game_id", "date"]], on="game_id").sort_values(["season", "team_id", "date"])
    by_season = {s: d for s, d in x.groupby("season")}

    def prior_fn(season, teams, prev):
        p_ids, p_off, p_def, _, _ = prev
        lo = dict(zip(p_ids, 100 * p_off))
        ld = dict(zip(p_ids, 100 * p_def))
        new_o = np.quantile(100 * p_off, new_team_pct)
        new_d = np.quantile(100 * p_def, 1 - new_team_pct)
        base = pd.DataFrame({"team_id": teams, "o_last": [lo.get(t, new_o) for t in teams],
                             "d_last": [ld.get(t, new_d) for t in teams]})
        team_o = team_models["o"].predict(base).to_numpy()
        team_d = team_models["d"].predict(base).to_numpy()
        sx = by_season.get(season)
        if sx is None:
            return lambda D: (team_o / 100, team_d / 100)
        tix = {t: i for i, t in enumerate(teams)}
        sx = sx[sx.team_id.isin(tix)].copy()
        sx = sx.merge(base, on="team_id", how="left")
        has = sx.roster_off.notna() & sx.roster_def.notna()
        po = np.full(len(sx), np.nan)
        pdf = np.full(len(sx), np.nan)
        if has.any():
            po[has.to_numpy()] = models["o"].predict(sx[has]).to_numpy()
            pdf[has.to_numpy()] = models["d"].predict(sx[has]).to_numpy()
        sx["po"], sx["pd"] = po, pdf
        # per-team arrays of (date, prior) to look up "as of" date D (first game on/after D)
        per_team = {}
        for t, d in sx.groupby("team_id"):
            per_team[tix[t]] = (d.date.to_numpy(), d.po.to_numpy(), d.pd.to_numpy())

        def at(D):
            o = team_o.copy()
            dd = team_d.copy()
            for i, (dates, a, b) in per_team.items():
                j = np.searchsorted(dates, D, side="left")
                j = min(j, len(dates) - 1)
                if np.isfinite(a[j]):
                    o[i], dd[i] = a[j], b[j]
            return o / 100, dd / 100
        return at
    return prior_fn
