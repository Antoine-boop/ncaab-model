"""
Player layer: player values -> roster-based team priors + availability (injury/absence) adjustments.

Player value (net points per 100 possessions) for player p on team T in season s:
    v_p = R_T + k * dev_p
where R_T is T's opponent-adjusted net rating at season end (from the ridge engine) and
dev_p is p's Hollinger game-score per 40 minutes relative to his team's minutes-weighted average,
shrunk toward zero for low-minute players. The minutes-weighted average of v_p over a roster
reproduces R_T, so values are on the team-rating scale and transfer across teams.

Walk-forward (no look-ahead) team features before each game:
    roster_off / roster_def / roster_net  minutes-share-weighted prior-season values of players
                                          already seen in this season's box scores
    seen_new_share                        minutes share of players with no prior D1 season
    ret_share                             share of last season's team minutes that has returned (seen)
    avail_adj                             expected-lineup value minus the decay-averaged lineup value
                                          the team rating was fit on (players absent last game -> out)
"""
from __future__ import annotations

import glob
import os
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")


def load_player_games(games: pd.DataFrame) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(RAW, "playerbox_*.parquet")))
    keep = ["game_id", "team_id", "athlete_id", "starter", "dnp", "minutes", "points", "fgm", "fga", "tpm", "tpa",
            "ftm", "fta", "offensiveRebounds", "defensiveRebounds", "assists", "steals", "blocks", "turnovers", "fouls"]
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        dfs.append(df[[c for c in keep if c in df.columns]])
    pb = pd.concat(dfs, ignore_index=True)
    pb = pb[pb.game_id.isin(games.game_id)]
    pb = pb.merge(games[["game_id", "season", "date"]], on="game_id")
    for c in keep[5:]:
        pb[c] = pd.to_numeric(pb[c], errors="coerce").fillna(0.0)
    pb = pb[pb.athlete_id.notna()]
    pb["athlete_id"] = pb.athlete_id.astype(np.int64)
    pb = pb.drop_duplicates(["game_id", "team_id", "athlete_id"])
    # Hollinger game score, split into offensive / defensive parts
    pb["gs_off"] = (pb.points + 0.4 * pb.fgm - 0.7 * pb.fga - 0.4 * (pb.fta - pb.ftm)
                    + 0.7 * pb.offensiveRebounds + 0.7 * pb.assists - pb.turnovers)
    pb["gs_def"] = 0.3 * pb.defensiveRebounds + pb.steals + 0.7 * pb.blocks - 0.4 * pb.fouls
    tmin = pb.groupby(["game_id", "team_id"]).minutes.transform("sum")
    pb["team_min"] = tmin
    pb = pb[(tmin >= 150)]                        # drop team-games with broken minutes
    pb["share"] = pb.minutes / pb.team_min       # share of the team's minutes (sums to 1 per team-game)
    return pb.sort_values(["date", "game_id"]).reset_index(drop=True)


def player_season_values(pb: pd.DataFrame, team_final: dict, k_off=1.0, k_def=1.0, m0=150.0) -> pd.DataFrame:
    """
    Season-level player values. team_final[season] = DataFrame(team_id, o, d) in points/100 poss
    (o: points scored above avg, d: points allowed above avg).
    """
    agg = pb.groupby(["season", "team_id", "athlete_id"]).agg(
        minutes=("minutes", "sum"), gs_off=("gs_off", "sum"), gs_def=("gs_def", "sum"),
        games=("minutes", lambda x: (x > 0).sum())).reset_index()
    tm = pb.groupby(["season", "team_id"]).agg(t_min=("minutes", "sum"), t_off=("gs_off", "sum"),
                                                t_def=("gs_def", "sum")).reset_index()
    agg = agg.merge(tm, on=["season", "team_id"])
    agg = agg[agg.minutes > 0]
    agg["min_share"] = agg.minutes / agg.t_min                  # share of team minutes (sums to 1)
    agg["off40"] = 40 * agg.gs_off / agg.minutes
    agg["def40"] = 40 * agg.gs_def / agg.minutes
    agg["toff40"] = 40 * agg.t_off / agg.t_min
    agg["tdef40"] = 40 * agg.t_def / agg.t_min
    shrink = agg.minutes / (agg.minutes + m0)
    agg["dev_off"] = (agg.off40 - agg.toff40) * shrink
    agg["dev_def"] = (agg.def40 - agg.tdef40) * shrink
    tf = pd.concat([df.assign(season=s) for s, df in team_final.items()], ignore_index=True)
    agg = agg.merge(tf, on=["season", "team_id"], how="left")
    agg["v_off"] = agg.o + k_off * agg.dev_off
    agg["v_def"] = agg.d - k_def * agg.dev_def                 # good defensive box stats lower points allowed
    return agg


# P(regular plays next game | consecutive team games missed), measured on 2016-22 box scores.
# Using it (p_return=P_RETURN) did not improve validation results (experiments/LOG.md E05); default is the
# simpler "missed the last game -> out" rule.
P_RETURN = (0.98, 0.48, 0.35, 0.23, 0.19, 0.13, 0.05)


def team_features(pb: pd.DataFrame, games: pd.DataFrame, pvals: pd.DataFrame, tau=45.0,
                  k_avail_off=1.0, k_avail_def=1.0, upcoming: pd.DataFrame | None = None,
                  p_return=None) -> pd.DataFrame:
    """
    Walk-forward per (game_id, team_id) pre-game features. pvals = player_season_values(...) for all seasons.
    upcoming: optional unplayed games (game_id, season, date, home_id, away_id); features come from each
    team's latest state.
    """
    pv = pvals.copy()
    pv["w"] = pv.minutes
    prev = (pv.assign(wo=pv.v_off * pv.w, wd=pv.v_def * pv.w)
            .groupby(["season", "athlete_id"]).agg(wo=("wo", "sum"), wd=("wd", "sum"), w=("w", "sum"))).reset_index()
    prev_map = {(s + 1, a): (o / w, d / w) for s, a, o, d, w in
                zip(prev.season, prev.athlete_id, prev.wo, prev.wd, prev.w)}
    lt = pvals[["season", "team_id", "athlete_id", "min_share"]]
    last_team_share = {(s + 1, t, a): m for s, t, a, m in zip(lt.season, lt.team_id, lt.athlete_id, lt.min_share)}

    up_by_team = defaultdict(list)
    if upcoming is not None and len(upcoming):
        for gid, s, D, h, a in zip(upcoming.game_id, upcoming.season, upcoming.date, upcoming.home_id, upcoming.away_id):
            up_by_team[(s, h)].append((D, gid))
            up_by_team[(s, a)].append((D, gid))

    out = []
    pb = pb.sort_values(["season", "team_id", "date", "game_id"])
    groups = {k: grp for k, grp in pb.groupby(["season", "team_id"], sort=False)}
    for key in set(groups) | set(up_by_team):
        s, t = key
        grp = groups.get(key)
        st = dict(cum_min=defaultdict(float), cum_off=defaultdict(float), cum_def=defaultdict(float),
                  dec_share=defaultdict(float), dec_h_share=defaultdict(float), dec_h_cnt=defaultdict(float),
                  dec_g=0.0, last_date=None, last_players=set(), n_games=0, streak=defaultdict(int))

        def advance(D):
            if st["last_date"] is not None:
                f = np.exp(-((D - st["last_date"]) / np.timedelta64(1, "D")) / tau)
                st["dec_g"] *= f
                for p in st["dec_share"]:
                    st["dec_share"][p] *= f
                    st["dec_h_share"][p] *= f
                    st["dec_h_cnt"][p] *= f
                st["last_date"] = D

        def features(gid):
            feat = dict(game_id=gid, team_id=t, season=s, n_games=st["n_games"])
            cum_min = st["cum_min"]
            seen = [p for p in cum_min if cum_min[p] > 0]
            if not seen:
                return feat
            tot = sum(cum_min[p] for p in seen)
            r_off = r_def = new_share = ret = 0.0
            for p in seen:
                sh = cum_min[p] / tot
                pv_ = prev_map.get((s, p))
                if pv_ is None:
                    new_share += sh
                else:
                    r_off += sh * pv_[0]
                    r_def += sh * pv_[1]
                ret += last_team_share.get((s, t, p), 0.0)
            known = 1 - new_share
            feat.update(roster_off=r_off / known if known > 1e-9 else np.nan,
                        roster_def=r_def / known if known > 1e-9 else np.nan,
                        seen_new_share=new_share, ret_share=ret)
            dec_share, dec_g = st["dec_share"], st["dec_g"]
            exp_sh, avg_sh = {}, {}
            for p in dec_share:
                avg_sh[p] = dec_share[p] / dec_g if dec_g > 0 else 0.0
                healthy = st["dec_h_share"][p] / st["dec_h_cnt"][p] if st["dec_h_cnt"][p] > 1e-9 else 0.0
                if p_return is None:
                    exp_sh[p] = healthy if p in st["last_players"] else 0.0
                else:
                    exp_sh[p] = healthy * p_return[min(st["streak"][p], len(p_return) - 1)]
            z = sum(exp_sh.values())
            if z > 0:
                a_off = a_def = 0.0
                for p in dec_share:
                    dlt = exp_sh[p] / z - avg_sh[p]
                    if dlt == 0:
                        continue
                    m = cum_min[p]
                    shrink = m / (m + 150.0)
                    a_off += dlt * (40 * st["cum_off"][p] / m) * shrink if m > 0 else 0.0
                    a_def += dlt * (40 * st["cum_def"][p] / m) * shrink if m > 0 else 0.0
                feat.update(avail_off=k_avail_off * a_off, avail_def=-k_avail_def * a_def,
                            missing_last=sum(avg_sh[p] for p in dec_share if p not in st["last_players"]))
            return feat

        if grp is not None:
            g_dates = grp.drop_duplicates("game_id")[["game_id", "date"]].to_numpy()
            rows_by_game = {gid: sub for gid, sub in grp.groupby("game_id", sort=False)}
            for gid, D in g_dates:
                if st["last_date"] is None:
                    st["last_date"] = D
                advance(D)
                out.append(features(gid))
                sub = rows_by_game[gid]
                played = sub[sub.minutes > 0]
                st["dec_g"] += 1.0
                st["last_players"] = set(played.athlete_id)
                for p in list(st["dec_share"].keys()):
                    st["streak"][p] = 0 if p in st["last_players"] else st["streak"][p] + 1
                for p, mi, sh, go, gdf in zip(played.athlete_id, played.minutes, played.share, played.gs_off,
                                              played.gs_def):
                    st["cum_min"][p] += mi
                    st["cum_off"][p] += go
                    st["cum_def"][p] += gdf
                    st["dec_share"][p] += sh
                    st["dec_h_share"][p] += sh
                    st["dec_h_cnt"][p] += 1.0
                st["n_games"] += 1
                st["last_date"] = D
        for D, gid in sorted(up_by_team.get(key, [])):
            if st["last_date"] is not None and D > st["last_date"]:
                advance(D)
            out.append(features(gid))
    return pd.DataFrame(out)
