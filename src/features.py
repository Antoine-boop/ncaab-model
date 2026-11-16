"""
Engineered stage-2 features validated in experiments/LOG.md (E01, E04, E06, E07, E08).
Everything is computed from information available before tip-off; upcoming (unplayed) games are
supported (their own results are NaN and never used).

  mkt_margin, mkt_total   market-derived ratings from previous games' closing lines        (E01)
  rest/b2b/travel/tz/elev schedule & travel context (src/context.py)                        (E04)
  hca_team, arena_tot     team home-court edge / arena scoring, prior 3 seasons, shrunk      (E06)
  conf_m, conf_t          conference residuals, walk-forward within season, shrunk           (E07)
  form_m, form_t          exponentially weighted recent residuals (half-life 6 games)        (E08)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import context as C

MKT_PARAMS = dict(lam=0.3, tau=1000.0, rho=0.7)
K_HCA, K_ARENA, K_CONF = 40.0, 40.0, 30.0
FORM_HALFLIFE = 6


def market_points(allg: pd.DataFrame) -> np.ndarray:
    """Implied points per team from closing lines, laid out like team_game_rows (home rows, then away rows)."""
    L, T = allg.close_spread.to_numpy(float), allg.close_total.to_numpy(float)
    return np.concatenate([(T - L) / 2, (T + L) / 2])


def add_team_hca(G: pd.DataFrame) -> pd.DataFrame:
    G = G.copy()
    e_m, e_t = G.margin - G.eff_margin, G.total_pts - G.eff_total
    G["hca_team"], G["arena_tot"] = 0.0, 0.0
    for s in sorted(G.season.unique()):
        h = G[(G.season < s) & (G.season >= s - 3) & ~G.neutral & e_m.notna()]
        em, et = e_m[h.index], e_t[h.index]
        home = pd.DataFrame({"t": h.home_id, "em": em, "et": et}).groupby("t").agg(sm=("em", "sum"), n=("em", "size"),
                                                                                    st=("et", "sum"))
        away = pd.DataFrame({"t": h.away_id, "em": -em}).groupby("t").agg(sm=("em", "sum"), n=("em", "size"))
        teams = home.index.union(away.index)
        hn = home.n.reindex(teams).fillna(0)
        hdev = 0.5 * (home.sm.reindex(teams).fillna(0) / (hn + K_HCA)
                      - away.sm.reindex(teams).fillna(0) / (away.n.reindex(teams).fillna(0) + K_HCA))
        tdev = home.st.reindex(teams).fillna(0) / (hn + K_ARENA)
        m = (G.season == s) & ~G.neutral
        G.loc[m, "hca_team"] = (G.loc[m, "home_id"].map(hdev).fillna(0).to_numpy()
                                + G.loc[m, "away_id"].map(hdev).fillna(0).to_numpy())
        G.loc[m, "arena_tot"] = G.loc[m, "home_id"].map(tdev).fillna(0).to_numpy()
    return G


def add_conference(G: pd.DataFrame) -> pd.DataFrame:
    G = G.copy()
    G["conf_m"], G["conf_t"] = 0.0, 0.0
    e_m, e_t = G.margin - G.eff_margin, G.total_pts - G.eff_total
    for s, d in G.groupby("season"):
        done = d[e_m[d.index].notna()]
        nonconf = done[~done.conf_game & (done.home_conf != done.away_conf)]
        nc = pd.concat([pd.DataFrame({"date": nonconf.date, "c": nonconf.home_conf, "e": e_m[nonconf.index]}),
                        pd.DataFrame({"date": nonconf.date, "c": nonconf.away_conf, "e": -e_m[nonconf.index]})])
        conf = done[done.conf_game]
        ct = pd.DataFrame({"date": conf.date, "c": conf.home_conf, "e": e_t[conf.index]})
        for D in np.sort(d.date.unique()):
            a = nc[nc.date < D].groupby("c").e.agg(["sum", "size"])
            cm = a["sum"] / (a["size"] + K_CONF)
            b = ct[ct.date < D].groupby("c").e.agg(["sum", "size"])
            tm = b["sum"] / (b["size"] + K_CONF)
            idx = d.index[d.date == D]
            G.loc[idx, "conf_m"] = (G.loc[idx, "home_conf"].map(cm).fillna(0)
                                    - G.loc[idx, "away_conf"].map(cm).fillna(0)).to_numpy()
            G.loc[idx, "conf_t"] = np.where(G.loc[idx, "conf_game"].to_numpy(),
                                            G.loc[idx, "home_conf"].map(tm).fillna(0).to_numpy(), 0.0)
    return G


def add_form(G: pd.DataFrame) -> pd.DataFrame:
    G = G.copy()
    e_m, e_t = G.margin - G.eff_margin, G.total_pts - G.eff_total
    long = pd.concat([
        pd.DataFrame({"game_id": G.game_id, "season": G.season, "date": G.date, "t": G.home_id, "em": e_m, "et": e_t}),
        pd.DataFrame({"game_id": G.game_id, "season": G.season, "date": G.date, "t": G.away_id, "em": -e_m, "et": e_t}),
    ]).sort_values(["t", "date", "game_id"])
    grp = long.groupby(["season", "t"])
    # residual of games already played only: shift so a game never sees itself; NaN (unplayed) rows are skipped
    long["fm"] = grp.em.transform(lambda x: x.shift().ewm(halflife=FORM_HALFLIFE, ignore_na=True).mean())
    long["ft"] = grp.et.transform(lambda x: x.shift().ewm(halflife=FORM_HALFLIFE, ignore_na=True).mean())
    long[["fm", "ft"]] = long.groupby(["season", "t"])[["fm", "ft"]].ffill().fillna(0.0)
    key = long.set_index(["game_id", "t"])
    for s, col in (("h", "home_id"), ("a", "away_id")):
        idx = pd.MultiIndex.from_arrays([G.game_id, G[col]])
        G[f"{s}_fm"] = key.fm.reindex(idx).to_numpy()
        G[f"{s}_ft"] = key.ft.reindex(idx).to_numpy()
    G["form_m"] = G.h_fm - G.a_fm
    G["form_t"] = G.h_ft + G.a_ft
    return G


def add_engineered(G: pd.DataFrame, games_all: pd.DataFrame) -> pd.DataFrame:
    G = C.add_context(G, games_all)
    G = add_team_hca(G)
    G = add_conference(G)
    G = add_form(G)
    return G
