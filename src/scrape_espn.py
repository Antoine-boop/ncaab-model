"""
ESPN data scraper for NCAA men's basketball (Division I).

Pulls, per season:
  * games      - scoreboard rows (teams, scores, neutral site, conference game, OT periods, ...)
  * teambox    - team box score totals (FGM/FGA, 3PM/3PA, FTM/FTA, OREB, DREB, TOV, ...)
  * playerbox  - player box score rows (minutes + counting stats)
  * odds       - every sportsbook ESPN archives for the game, with open/close where available

Seasons are labelled by their ending year (2025 = 2024-25), matching ESPN.

Usage:
    python src/scrape_espn.py 2016 2017 ... [--refresh]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import os
import random
import sys
import threading
import time

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")

SB_URL = ("https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/"
          "scoreboard?dates={d}&groups=50&limit=1000")
SUM_URL = ("https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/"
           "summary?event={e}")
ODDS_URL = ("https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/"
            "events/{e}/competitions/{e}/odds?lang=en&region=us&limit=100")

_local = threading.local()


def _session() -> requests.Session:
    s = getattr(_local, "s", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": "Mozilla/5.0 (research script)", "Accept-Encoding": "gzip"})
        _local.s = s
    return s


def get_json(url: str, tries: int = 6):
    delay = 1.0
    for i in range(tries):
        try:
            r = _session().get(url, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            # 429 / 5xx -> back off
        except Exception:
            pass
        time.sleep(delay + random.random())
        delay = min(delay * 2, 30)
    return None


# ----------------------------------------------------------------------------------------------
# parsing helpers
# ----------------------------------------------------------------------------------------------

def _num(x):
    """Parse ESPN odds/line strings: '+4.5', '-110', 'EVEN', 'PK', 1.5, None."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().upper()
    if s in ("", "OFF", "N/A", "NA", "-"):
        return None
    if s in ("EVEN", "EV"):
        return 100.0
    if s in ("PK", "PICK", "PICKEM"):
        return 0.0
    try:
        return float(s.replace("+", "").replace(",", ""))
    except ValueError:
        return None


def _dig(d, *path):
    for p in path:
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


def _split_made_att(s):
    try:
        a, b = str(s).split("-")
        return float(a), float(b)
    except Exception:
        return None, None


def parse_scoreboard(sb: dict, season: int) -> list[dict]:
    rows = []
    for e in sb.get("events", []):
        c = e["competitions"][0]
        comps = {t["homeAway"]: t for t in c["competitors"]}
        if "home" not in comps or "away" not in comps:
            continue
        h, a = comps["home"], comps["away"]
        st = c["status"]["type"]
        notes = c.get("notes") or []
        rows.append(dict(
            game_id=int(e["id"]),
            season=season,
            date_utc=e["date"],
            season_type=_dig(e, "season", "type"),
            neutral=bool(c.get("neutralSite")),
            conf_game=bool(c.get("conferenceCompetition")),
            completed=bool(st.get("completed")),
            status=st.get("name"),
            home_id=int(h["team"]["id"]), away_id=int(a["team"]["id"]),
            home_name=h["team"].get("displayName"), away_name=a["team"].get("displayName"),
            home_abbr=h["team"].get("abbreviation"), away_abbr=a["team"].get("abbreviation"),
            home_conf=h["team"].get("conferenceId"), away_conf=a["team"].get("conferenceId"),
            home_score=_num(h.get("score")), away_score=_num(a.get("score")),
            n_periods=max(len(h.get("linescores") or []), len(a.get("linescores") or [])),
            venue_id=_dig(c, "venue", "id"),
            venue_city=_dig(c, "venue", "address", "city"),
            venue_state=_dig(c, "venue", "address", "state"),
            attendance=c.get("attendance"),
            note=notes[0].get("headline") if notes else None,
        ))
    return rows


TEAM_STAT_MAP = {
    "fieldGoalsMade-fieldGoalsAttempted": ("fgm", "fga"),
    "threePointFieldGoalsMade-threePointFieldGoalsAttempted": ("tpm", "tpa"),
    "freeThrowsMade-freeThrowsAttempted": ("ftm", "fta"),
}
TEAM_STAT_SINGLE = {
    "totalRebounds": "treb", "offensiveRebounds": "oreb", "defensiveRebounds": "dreb",
    "assists": "ast", "steals": "stl", "blocks": "blk", "turnovers": "tov_player",
    "teamTurnovers": "tov_team", "totalTurnovers": "tov", "fouls": "pf",
    "technicalFouls": "tech", "flagrantFouls": "flagrant", "largestLead": "largest_lead",
    "fastBreakPoints": "fb_pts", "pointsInPaint": "paint_pts", "turnoverPoints": "tov_pts",
}


def parse_summary(d: dict, game_id: int):
    """Return (team_rows, player_rows, extra_game_info)."""
    team_rows, player_rows = [], []
    if not d:
        return team_rows, player_rows, {}
    comp = _dig(d, "header", "competitions") or [{}]
    comp = comp[0]
    side = {}
    for t in comp.get("competitors", []):
        side[str(t["id"])] = t.get("homeAway")
    for t in _dig(d, "boxscore", "teams") or []:
        tid = str(t["team"]["id"])
        row = dict(game_id=game_id, team_id=int(tid), home_away=side.get(tid))
        for s in t.get("statistics", []):
            n = s.get("name")
            if n in TEAM_STAT_MAP:
                m, a = _split_made_att(s.get("displayValue"))
                row[TEAM_STAT_MAP[n][0]], row[TEAM_STAT_MAP[n][1]] = m, a
            elif n in TEAM_STAT_SINGLE:
                row[TEAM_STAT_SINGLE[n]] = _num(s.get("displayValue"))
        team_rows.append(row)
    for t in _dig(d, "boxscore", "players") or []:
        tid = int(t["team"]["id"])
        for blk in t.get("statistics", []):
            keys = blk.get("keys") or []
            for ath in blk.get("athletes", []):
                a = ath.get("athlete") or {}
                stats = ath.get("stats") or []
                r = dict(game_id=game_id, team_id=tid, athlete_id=_num(a.get("id")),
                         name=a.get("displayName"), pos=_dig(a, "position", "abbreviation"),
                         starter=bool(ath.get("starter")), dnp=bool(ath.get("didNotPlay")))
                for k, v in zip(keys, stats):
                    if "-" in k and k.count("-") == 1 and "Made" in k:
                        m, att = _split_made_att(v)
                        base = {"fieldGoalsMade-fieldGoalsAttempted": ("fgm", "fga"),
                                "threePointFieldGoalsMade-threePointFieldGoalsAttempted": ("tpm", "tpa"),
                                "freeThrowsMade-freeThrowsAttempted": ("ftm", "fta")}.get(k)
                        if base:
                            r[base[0]], r[base[1]] = m, att
                    else:
                        r[k] = _num(v)
                player_rows.append(r)
    gi = d.get("gameInfo") or {}
    extra = dict(game_id=game_id, attendance_sum=gi.get("attendance"),
                 n_officials=len(gi.get("officials") or []),
                 officials="|".join(o.get("displayName", "") for o in (gi.get("officials") or [])))
    return team_rows, player_rows, extra


def _odds_side(o, which):
    """Extract line/price from a team-odds dict's open/close/current block."""
    blk = (o or {}).get(which) or {}
    return (_num(_dig(blk, "pointSpread", "american")), _num(_dig(blk, "spread", "american")),
            _num(_dig(blk, "moneyLine", "american")))


def parse_odds(d: dict, game_id: int) -> list[dict]:
    rows = []
    if not d:
        return rows
    for it in d.get("items", []):
        prov = it.get("provider") or {}
        h, a = it.get("homeTeamOdds") or {}, it.get("awayTeamOdds") or {}
        r = dict(game_id=game_id, provider_id=_num(prov.get("id")), provider=prov.get("name"),
                 spread=_num(it.get("spread")), total=_num(it.get("overUnder")),
                 over_odds=_num(it.get("overOdds")), under_odds=_num(it.get("underOdds")),
                 home_ml=_num(h.get("moneyLine")), away_ml=_num(a.get("moneyLine")),
                 home_spread_odds=_num(h.get("spreadOdds")), away_spread_odds=_num(a.get("spreadOdds")),
                 details=it.get("details"))
        for which in ("open", "close"):
            hs, hso, hml = _odds_side(h, which)
            as_, aso, aml = _odds_side(a, which)
            top = it.get(which) or {}
            r.update({
                f"{which}_home_spread": hs, f"{which}_away_spread": as_,
                f"{which}_home_spread_odds": hso, f"{which}_away_spread_odds": aso,
                f"{which}_home_ml": hml, f"{which}_away_ml": aml,
                f"{which}_total": _num(_dig(top, "total", "american")),
                f"{which}_over_odds": _num(_dig(top, "over", "american")),
                f"{which}_under_odds": _num(_dig(top, "under", "american")),
            })
        rows.append(r)
    return rows


# ----------------------------------------------------------------------------------------------
# season drivers
# ----------------------------------------------------------------------------------------------

def season_dates(season: int, end: dt.date | None = None):
    start = dt.date(season - 1, 11, 1)
    stop = dt.date(season, 4, 10)
    if end is not None:
        stop = min(stop, end)
    d = start
    while d <= stop:
        yield d
        d += dt.timedelta(days=1)


def scrape_season(season: int, workers: int = 16, refresh: bool = False, end: dt.date | None = None):
    out = {k: os.path.join(RAW, f"{k}_{season}.parquet") for k in ("games", "teambox", "playerbox", "odds", "gameinfo")}
    if not refresh and all(os.path.exists(p) for p in out.values()):
        print(f"[{season}] already scraped, skipping", flush=True)
        return
    t0 = time.time()
    dates = list(season_dates(season, end))
    with cf.ThreadPoolExecutor(workers) as ex:
        sbs = list(ex.map(lambda d: (d, get_json(SB_URL.format(d=d.strftime("%Y%m%d")))), dates))
    games = []
    for d, sb in sbs:
        if sb is None:
            print(f"[{season}] WARN scoreboard missing {d}", flush=True)
            continue
        games.extend(parse_scoreboard(sb, season))
    games = pd.DataFrame(games).drop_duplicates("game_id")
    print(f"[{season}] {len(games)} games from {len(dates)} scoreboards ({time.time()-t0:.0f}s)", flush=True)

    done_ids = games.loc[games.completed, "game_id"].tolist()
    all_ids = games["game_id"].tolist()

    team_rows, player_rows, extras, odds_rows = [], [], [], []
    lock = threading.Lock()

    def do_summary(gid):
        tr, pr, ex_ = parse_summary(get_json(SUM_URL.format(e=gid)), gid)
        with lock:
            team_rows.extend(tr); player_rows.extend(pr)
            if ex_:
                extras.append(ex_)

    def do_odds(gid):
        rows = parse_odds(get_json(ODDS_URL.format(e=gid)), gid)
        with lock:
            odds_rows.extend(rows)

    with cf.ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(do_summary, g) for g in done_ids] + [ex.submit(do_odds, g) for g in all_ids]
        for i, f in enumerate(cf.as_completed(futs)):
            f.result()
            if (i + 1) % 2000 == 0:
                print(f"[{season}] {i+1}/{len(futs)} requests ({time.time()-t0:.0f}s)", flush=True)

    games.to_parquet(out["games"], index=False)
    pd.DataFrame(team_rows).to_parquet(out["teambox"], index=False)
    pd.DataFrame(player_rows).to_parquet(out["playerbox"], index=False)
    pd.DataFrame(extras).to_parquet(out["gameinfo"], index=False)
    pd.DataFrame(odds_rows).to_parquet(out["odds"], index=False)
    print(f"[{season}] done: games={len(games)} teambox={len(team_rows)} players={len(player_rows)} "
          f"odds_rows={len(odds_rows)} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seasons", nargs="+", type=int)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--end", type=str, default=None, help="YYYY-MM-DD last date to pull (current season)")
    args = ap.parse_args()
    end = dt.date.fromisoformat(args.end) if args.end else None
    os.makedirs(RAW, exist_ok=True)
    for s in args.seasons:
        scrape_season(s, workers=args.workers, refresh=args.refresh, end=end)
