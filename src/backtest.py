"""
Betting backtest for spreads, totals and moneylines against opening and closing lines.

Calibration (fit on DEV seasons only):
  spread : P(home covers)  = logistic(a0 + a1 * (pred_margin + line) / sig_m)
  total  : P(over)          = logistic(b0 + b1 * (pred_total - line) / sig_t)
  ML     : logit P(home win) = c0 + c1 * logit(p_mkt_novig) + c2 * (logit(p_model) - logit(p_mkt_novig))
Bets: one side per game per market when EV >= threshold, flat 1 unit risked, settled at the price we
would have received (book juice when archived, -110 otherwise).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_dataset import american_to_decimal, american_to_prob  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = list(range(2016, 2023))


def sig_margin(spread):
    """Residual SD of the margin around the line grows in mismatches (fit on 2016-22 closes)."""
    return 11.0 + 0.11 * np.maximum(0.0, np.abs(np.asarray(spread, float)) - 12.0)


def sig_total(total):
    """Residual SD of the total grows with the total (fit on 2016-22 closes)."""
    return 16.6 + 0.07 * (np.asarray(total, float) - 140.0)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def market_frame(G: pd.DataFrame, line_type: str) -> pd.DataFrame:
    """Unified columns for one line type: spread, total, ml prices and juice."""
    M = pd.DataFrame(index=G.index)
    if line_type == "close":
        sp = G.pb_spread.where(G.pb_spread.notna(), (G.close_spread * 2).round() / 2)
        tot = G.pb_total.where(G.pb_total.notna(), (G.close_total * 2).round() / 2)
        M["spread"], M["total"] = sp, tot
        M["h_sp_odds"] = G.pb_home_spread_odds.where(G.pb_spread.notna())
        M["a_sp_odds"] = G.pb_away_spread_odds.where(G.pb_spread.notna())
        M["o_odds"] = G.pb_over_odds.where(G.pb_total.notna())
        M["u_odds"] = G.pb_under_odds.where(G.pb_total.notna())
        hml = G.pb_home_ml.where(G.pb_home_ml.notna() & G.pb_away_ml.notna(), G.close_home_ml)
        aml = G.pb_away_ml.where(G.pb_home_ml.notna() & G.pb_away_ml.notna(), G.close_away_ml)
    else:
        # prefer the primary book's own opener (bettable as-is); fall back to cross-book median
        has_pb_sp = G.pb_open_home_spread.notna()
        has_pb_t = G.pb_open_total.notna()
        M["spread"] = G.pb_open_home_spread.where(has_pb_sp, (G.open_spread * 2).round() / 2)
        M["total"] = G.pb_open_total.where(has_pb_t, (G.open_total * 2).round() / 2)
        M["h_sp_odds"] = G.pb_open_home_spread_odds.where(has_pb_sp)
        M["a_sp_odds"] = G.pb_open_away_spread_odds.where(has_pb_sp)
        M["o_odds"] = G.pb_open_over_odds.where(has_pb_t)
        M["u_odds"] = G.pb_open_under_odds.where(has_pb_t)
        has_pb_ml = G.pb_open_home_ml.notna() & G.pb_open_away_ml.notna()
        hml = G.pb_open_home_ml.where(has_pb_ml, G.open_home_ml)
        aml = G.pb_open_away_ml.where(has_pb_ml, G.open_away_ml)
    for c in ("h_sp_odds", "a_sp_odds", "o_odds", "u_odds"):
        M[c] = M[c].fillna(-110.0)
        M.loc[M[c].abs() < 100, c] = -110.0
    M["h_ml"], M["a_ml"] = hml, aml
    # sanity: the two moneyline prices must form a sane book and agree with the same book's spread
    ph, pa = american_to_prob(M.h_ml), american_to_prob(M.a_ml)
    over = ph + pa
    p_ml = ph / over
    p_sp = norm.cdf(-M.spread.to_numpy(float) / 11.0)
    with np.errstate(invalid="ignore"):
        bad = ((over < 1.0) | (over > 1.12) | (np.abs(p_ml - p_sp) > 0.12) | (M.h_ml.abs().to_numpy() < 100)
               | (M.a_ml.abs().to_numpy() < 100) | ~np.isfinite(p_ml) | ~np.isfinite(p_sp))
    M.loc[bad, ["h_ml", "a_ml"]] = np.nan
    return M


def fit_calibration(G, M, sig_m, sig_t, fit_mask):
    cal = {}
    d = G[fit_mask].join(M[fit_mask])
    # spread
    s = d.dropna(subset=["spread", "pred_margin"])
    x = s.margin + s.spread
    s = s[x != 0]
    y = (s.margin + s.spread > 0).astype(float)
    z = (s.pred_margin + s.spread) / sig_margin(s.spread)
    cal["spread"] = sm.Logit(y, sm.add_constant(z)).fit(disp=0).params.to_numpy()
    # total
    t = d.dropna(subset=["total", "pred_total"])
    t = t[t.total_pts != t.total]
    y = (t.total_pts > t.total).astype(float)
    z = (t.pred_total - t.total) / sig_total(t.total)
    cal["total"] = sm.Logit(y, sm.add_constant(z)).fit(disp=0).params.to_numpy()
    # moneyline
    m = d.dropna(subset=["h_ml", "a_ml", "p_home"])
    ph, pa = american_to_prob(m.h_ml), american_to_prob(m.a_ml)
    pm = ph / (ph + pa)
    X = np.column_stack([np.ones(len(m)), _logit(pm), _logit(m.p_home) - _logit(pm)])
    y = (m.margin > 0).astype(float)
    cal["ml"] = sm.Logit(y, X).fit(disp=0).params
    # spread-derived ML: market margin shifted toward the model by kappa; residual sd around it
    k = d.dropna(subset=["spread", "pred_margin"])
    kap = sm.OLS(k.margin + k.spread, k.pred_margin + k.spread).fit().params.iloc[0]
    mu = -k.spread + kap * (k.pred_margin + k.spread)
    cal["ml_kappa"] = np.array([float(kap), float(np.sqrt(np.mean((k.margin - mu) ** 2)))])
    return cal


def make_bets(G, M, cal, sig_m, sig_t, thr=dict(spread=0.03, total=0.03, ml=0.03), ml_prob_range=(0.15, 0.85)):
    d = G.join(M)
    out = []
    # ---- spreads ----
    s = d.dropna(subset=["spread", "pred_margin"])
    a = cal["spread"]
    p_home = _sigmoid(a[0] + a[1] * (s.pred_margin + s.spread) / sig_margin(s.spread))
    dec_h, dec_a = american_to_decimal(s.h_sp_odds), american_to_decimal(s.a_sp_odds)
    ev_h = p_home * (dec_h - 1) - (1 - p_home)
    ev_a = (1 - p_home) * (dec_a - 1) - p_home
    side = np.where(ev_h >= ev_a, 1, -1)
    ev = np.maximum(ev_h, ev_a)
    x = s.margin + s.spread
    win = np.where(side == 1, x > 0, x < 0)
    push = x == 0
    dec = np.where(side == 1, dec_h, dec_a)
    out.append(pd.DataFrame(dict(game_id=s.game_id, season=s.season, date=s.date, market="spread", side=side,
                                 line=s.spread, ev=ev, p=np.where(side == 1, p_home, 1 - p_home), dec=dec,
                                 profit=np.where(push, 0.0, np.where(win, dec - 1, -1.0)),
                                 edge_pts=(s.pred_margin + s.spread) * side), index=s.index))
    # ---- totals ----
    t = d.dropna(subset=["total", "pred_total"])
    b = cal["total"]
    p_over = _sigmoid(b[0] + b[1] * (t.pred_total - t.total) / sig_total(t.total))
    dec_o, dec_u = american_to_decimal(t.o_odds), american_to_decimal(t.u_odds)
    ev_o = p_over * (dec_o - 1) - (1 - p_over)
    ev_u = (1 - p_over) * (dec_u - 1) - p_over
    side = np.where(ev_o >= ev_u, 1, -1)
    ev = np.maximum(ev_o, ev_u)
    x = t.total_pts - t.total
    win = np.where(side == 1, x > 0, x < 0)
    push = x == 0
    dec = np.where(side == 1, dec_o, dec_u)
    out.append(pd.DataFrame(dict(game_id=t.game_id, season=t.season, date=t.date, market="total", side=side,
                                 line=t.total, ev=ev, p=np.where(side == 1, p_over, 1 - p_over), dec=dec,
                                 profit=np.where(push, 0.0, np.where(win, dec - 1, -1.0)),
                                 edge_pts=(t.pred_total - t.total) * side), index=t.index))
    # ---- moneyline ----
    m = d.dropna(subset=["h_ml", "a_ml", "p_home", "spread"])
    ph, pa = american_to_prob(m.h_ml), american_to_prob(m.a_ml)
    pm = ph / (ph + pa)
    if "ml_kappa" in cal:
        # anchor on the market's own no-vig win probability; shift it only by the model's calibrated
        # disagreement with the spread (so ML bets express the side opinion, not ML-vs-spread noise)
        kap, sig_r = cal["ml_kappa"]
        p = norm.cdf(norm.ppf(np.clip(pm, 1e-4, 1 - 1e-4)) + kap * (m.pred_margin + m.spread) / sig_margin(m.spread))
    else:
        c = np.asarray(cal["ml"])
        p = _sigmoid(c[0] + c[1] * _logit(pm) + c[2] * (_logit(m.p_home) - _logit(pm)))
    dec_h, dec_a = american_to_decimal(m.h_ml), american_to_decimal(m.a_ml)
    ev_h = p * (dec_h - 1) - (1 - p)
    ev_a = (1 - p) * (dec_a - 1) - p
    side = np.where(ev_h >= ev_a, 1, -1)
    ev = np.maximum(ev_h, ev_a)
    # risk control: no extreme long shots / heavy favourites (model tails are least reliable there)
    pm_side = np.where(side == 1, pm, 1 - pm)
    ev = np.where((pm_side >= ml_prob_range[0]) & (pm_side <= ml_prob_range[1]), ev, -np.inf)
    win = np.where(side == 1, m.margin > 0, m.margin < 0)
    dec = np.where(side == 1, dec_h, dec_a)
    out.append(pd.DataFrame(dict(game_id=m.game_id, season=m.season, date=m.date, market="ml", side=side,
                                 line=np.where(side == 1, m.h_ml, m.a_ml), ev=ev, p=np.where(side == 1, p, 1 - p),
                                 dec=dec, profit=np.where(win, dec - 1, -1.0),
                                 edge_pts=np.nan), index=m.index))
    B = pd.concat(out)
    B["thr"] = B.market.map(thr)
    return B


def summarize(B, by=("market", "season")):
    def f(x):
        n = len(x)
        return pd.Series(dict(n=n, win=(x.profit > 0).mean(), roi=x.profit.sum() / n if n else np.nan,
                              units=x.profit.sum(), avg_ev=x.ev.mean(),
                              z=(x.profit.mean() / (x.profit.std(ddof=1) / np.sqrt(n))) if n > 2 else np.nan))
    return B.groupby(list(by)).apply(f, include_groups=False)
