"""
Walk-forward opponent-adjusted ratings (ridge regression with recency decay + preseason priors).

For every stat "component" c and every game day D in a season, we fit

    y_i = mu + off[o_i] + def[d_i] + hca * loc_i + e_i,     weight w_i * exp(-(D - date_i)/tau)

on all games strictly before D, with a ridge prior pulling off/def toward the team's preseason
prior (last season's final rating x rho). Normal equations are maintained incrementally:
moving the evaluation day forward just multiplies the accumulated system by exp(-dt/tau).

Every (offense, defense) row on day D then gets a *pre-game* prediction - no look-ahead.

Components (offense perspective; the defense column is the opponent's effect):
    ppp   points / possession                  weight = poss
    tempo possessions per 40 min (shared team parameter, off+def)   weight = 1
    tov   turnovers / possession               weight = poss
    orb   offensive rebounds / missed FG       weight = misses
    ftr   FTA / FGA                            weight = fga
    ftp   FTM / FTA                            weight = fta
    r3    3PA / FGA                            weight = fga
    p3    3PM / 3PA                            weight = tpa
    p2    2PM / 2PA                            weight = twa
"""
from __future__ import annotations

import dataclasses
import numpy as np
import pandas as pd
import scipy.linalg as sla

COMPONENTS = {
    # name: (numerator col, weight col, shared parameters?)
    "ppp": ("pts", "poss", False),
    "tempo": ("tempo", None, True),
    "tov": ("tov", "poss", False),
    "orb": ("oreb", "miss", False),
    "ftr": ("fta", "fga", False),
    "ftp": ("ftm", "fta", False),
    "r3": ("tpa", "fga", False),
    "p3": ("tpm", "tpa", False),
    "p2": ("twm", "twa", False),
    # market-implied points from previous games' closing lines (top-down feature, experiments only)
    "mkt": ("mkt_pts", None, False),
    # log points per possession (multiplicative efficiency; experiments)
    "lppp": ("lppp_num", "poss", False),
}


@dataclasses.dataclass
class CompParams:
    lam_off: float          # prior strength (in units of the component weight) for offense ratings
    lam_def: float          # ... for defense ratings
    tau: float = 60.0       # recency decay time-constant in days
    rho_off: float = 0.6    # carry-over of last season's offense rating into the prior
    rho_def: float = 0.6
    lam_mu: float = 1e-2    # tiny: league mean is basically free
    lam_h: float = 1.0      # home-court prior strength (relative to weight units)
    new_team_q: float = 0.25  # prior quantile for teams new to D1


def component_arrays(tg: pd.DataFrame, comp: str):
    num, wcol, shared = COMPONENTS[comp]
    if wcol is None:
        # unweighted level component (tempo, market-implied points): weight 1 where observed
        y = tg[num].to_numpy(float)
        w = np.isfinite(y).astype(float)
        y = np.where(np.isfinite(y), y, 0.0)
    else:
        w = tg[wcol].to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            y = tg[num].to_numpy(float) / w
        bad = ~np.isfinite(y) | (w <= 0)
        y[bad] = 0.0
        w = np.where(bad, 0.0, w)
    return y, w, shared


class SeasonRidge:
    """Incremental weighted ridge for one component in one season."""

    def __init__(self, n_teams: int, shared: bool, params: CompParams, prior_mu: float, prior_h: float,
                 prior_off: np.ndarray, prior_def: np.ndarray | None):
        self.n, self.shared, self.p = n_teams, shared, params
        k = 2 + (n_teams if shared else 2 * n_teams)
        self.k = k
        self.A = np.zeros((k, k))
        self.b = np.zeros(k)
        lam = np.empty(k)
        lam[0], lam[1] = params.lam_mu, params.lam_h
        beta0 = np.empty(k)
        beta0[0], beta0[1] = prior_mu, prior_h
        if shared:
            lam[2:] = params.lam_off
            beta0[2:] = prior_off
        else:
            lam[2:2 + n_teams] = params.lam_off
            lam[2 + n_teams:] = params.lam_def
            beta0[2:2 + n_teams] = prior_off
            beta0[2 + n_teams:] = prior_def
        self.lam, self.beta0 = lam, beta0
        self.Lb0 = lam * beta0
        if not shared:
            self._o_idx = np.arange(2, 2 + n_teams)
            self._keep = np.r_[0, 1, np.arange(2 + n_teams, 2 + 2 * n_teams)]

    def _cols(self, off, de):
        if self.shared:
            return 2 + off, 2 + de
        return 2 + off, 2 + self.n + de

    def decay(self, days: float):
        f = np.exp(-days / self.p.tau)
        self.A *= f
        self.b *= f

    def add(self, off, de, loc, y, w):
        co, cd = self._cols(off, de)
        k = self.k
        # design rows have entries: [0]=1, [1]=loc, [co]=1, [cd]=1
        idx = np.stack([np.zeros_like(co), np.ones_like(co), co, cd], axis=1)
        val = np.stack([np.ones(len(co)), loc.astype(float), np.ones(len(co)), np.ones(len(co))], axis=1)
        wv = val * w[:, None]
        # A += sum_i w_i x_i x_i^T  (4x4 outer products scattered)
        for a in range(4):
            for c in range(4):
                np.add.at(self.A, (idx[:, a], idx[:, c]), wv[:, a] * val[:, c])
        for a in range(4):
            np.add.at(self.b, idx[:, a], wv[:, a] * y)

    def set_prior(self, prior_off: np.ndarray, prior_def: np.ndarray | None):
        """Replace team prior means (e.g. roster-based priors that update as rosters are observed)."""
        n = self.n
        if self.shared:
            self.beta0[2:] = prior_off
        else:
            self.beta0[2:2 + n] = prior_off
            self.beta0[2 + n:] = prior_def
        self.Lb0 = self.lam * self.beta0

    def solve(self):
        r = self.b + self.Lb0
        if self.shared:
            M = self.A + np.diag(self.lam)
            return sla.solve(M, r, assume_a="pos", check_finite=False)
        # Schur complement: the offense block of the normal matrix is diagonal (each row has exactly
        # one offense team), so eliminate it and solve the (n+2)-dim SPD system for [mu, h, def].
        n, k = self.n, self.k
        keep = self._keep
        o_idx = self._o_idx
        a_o = self.A[o_idx, o_idx] + self.lam[o_idx]
        M_ko = self.A[np.ix_(keep, o_idx)]
        Mk_scaled = M_ko / a_o
        S = self.A[np.ix_(keep, keep)] - Mk_scaled @ M_ko.T
        S[np.diag_indices_from(S)] += self.lam[keep]
        rhs = r[keep] - Mk_scaled @ r[o_idx]
        x = sla.solve(S, rhs, assume_a="pos", check_finite=False)
        beta = np.empty(k)
        beta[keep] = x
        beta[o_idx] = (r[o_idx] - M_ko.T @ x) / a_o
        return beta

    def predict(self, beta, off, de, loc):
        co, cd = self._cols(off, de)
        return beta[0] + beta[1] * loc + beta[co] + beta[cd]


def run_component(tg: pd.DataFrame, comp: str, params: CompParams, seasons: list[int],
                  burn_in_season: int | None = None, return_state: bool = False, prior_fn=None):
    """
    Walk forward through seasons; returns pre-game prediction for each row of tg (NaN for rows in
    the burn-in season) and the final ratings for each season (for priors / live use).

    tg must contain: season, date, off, def, loc + component columns.
    """
    y_all, w_all, shared = component_arrays(tg, comp)
    pred = np.full(len(tg), np.nan)
    final = {}
    prev = None  # (team_ids, off, def, mu, h)
    all_seasons = ([burn_in_season] if burn_in_season is not None else []) + list(seasons)
    for s in all_seasons:
        m = (tg.season.to_numpy() == s)
        idx_rows = np.flatnonzero(m)
        if len(idx_rows) == 0:
            continue
        sub = tg.iloc[idx_rows]
        teams = np.unique(np.concatenate([sub["off"].to_numpy(), sub["def"].to_numpy()]))
        tix = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        off = np.array([tix[t] for t in sub["off"].to_numpy()])
        de = np.array([tix[t] for t in sub["def"].to_numpy()])
        loc = sub["loc"].to_numpy(float)
        y, w = y_all[idx_rows], w_all[idx_rows]

        # priors
        if prev is None:
            wsum = w.sum()
            mu0 = float((y * w).sum() / max(wsum, 1e-9))
            h0 = 0.0
            po = np.zeros(n)
            pd_ = np.zeros(n)
        else:
            p_ids, p_off, p_def, mu0, h0 = prev
            pmap_o = dict(zip(p_ids, p_off))
            pmap_d = dict(zip(p_ids, p_def)) if p_def is not None else None
            q_o = np.quantile(p_off, params.new_team_q if comp != "tempo" else 0.5)
            po = np.array([pmap_o.get(t, q_o / max(params.rho_off, 1e-9)) for t in teams]) * params.rho_off
            if shared:
                pd_ = None
            else:
                # for defense, "bad" is a high value for most components -> use the opposite quantile
                q_d = np.quantile(p_def, 1 - params.new_team_q)
                pd_ = np.array([pmap_d.get(t, q_d / max(params.rho_def, 1e-9)) for t in teams]) * params.rho_def
        model = SeasonRidge(n, shared, params, mu0, h0, po, pd_)

        dates = sub["date"].to_numpy()
        udates, inv = np.unique(dates, return_inverse=True)
        last = None
        beta = None
        prior_ctx = None
        if prior_fn is not None and prev is not None and s != burn_in_season:
            prior_ctx = prior_fn(s, teams, prev)      # returns callable(date) -> (prior_off, prior_def) or None
        for di, D in enumerate(udates):
            if last is not None:
                model.decay((D - last) / np.timedelta64(1, "D"))
            rows = np.flatnonzero(inv == di)
            if prior_ctx is not None:
                pr = prior_ctx(D)
                if pr is not None:
                    model.set_prior(pr[0], pr[1])
            beta = model.solve()
            if s != burn_in_season:
                pred[idx_rows[rows]] = model.predict(beta, off[rows], de[rows], loc[rows])
            ok = w[rows] > 0
            r = rows[ok]
            model.add(off[r], de[r], loc[r], y[r], w[r])
            last = D
        beta = model.solve()
        if shared:
            f_off, f_def = beta[2:], None
        else:
            f_off, f_def = beta[2:2 + n], beta[2 + n:]
        prev = (teams, f_off, f_def, float(beta[0]), float(beta[1]))
        final[s] = dict(teams=teams, off=f_off, de=f_def, mu=float(beta[0]), h=float(beta[1]),
                        state=model if return_state else None, tix=tix)
    return pred, final


# --------------------------------------------------------------------------------------------
# Possession identity: components -> points per possession
# --------------------------------------------------------------------------------------------

def ppp_from_components(tov, orb, ftr, ftp, r3, p3, p2, ft_poss=0.475):
    """
    Exact identity for points per (own) possession, given
        t = TOV/P, orb = OREB/missed FG, ftr = FTA/FGA, ftp = FT%, r3 = 3PA/FGA, p3 = 3P%, p2 = 2P%
    with P = FGA - OREB + TOV + 0.475*FTA.
    """
    q = (1 - r3) * p2 + r3 * p3                      # FG%
    x = (1 - tov) / (1 - orb * (1 - q) + ft_poss * ftr)   # FGA per possession
    return x * (2 * (1 - r3) * p2 + 3 * r3 * p3 + ftr * ftp)
