"""
Assemble modelling tables from raw ESPN pulls.

Outputs (data/processed/):
  games.parquet   one row per completed D1-vs-D1 game with home/away box stats, possessions,
                  four-factor components and consolidated market lines (open + close).
  teamgame.parquet  two rows per game (offense perspective) - convenient for rating fits.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")

FT_POSS = 0.475  # FTA weight in possession estimate (KenPom convention)

# Providers that are not pre-game sportsbook prices
EXCLUDE_PROVIDERS = ("Live",)            # in-game feeds ("ESPN Bet - Live Odds", "DraftKings - Live Odds")
AGGREGATORS = {"numberfire", "teamrankings", "accuscore", "consensus", "Opening"}


def _read(kind: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(RAW, f"{kind}_*.parquet")))
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        if len(df):
            if "season" not in df.columns:
                df["season"] = int(os.path.basename(f).split("_")[1].split(".")[0])
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# ------------------------------------------------------------------------------------------
# market lines
# ------------------------------------------------------------------------------------------

def american_to_prob(ml):
    ml = np.asarray(ml, dtype=float)
    return np.where(ml < 0, -ml / (-ml + 100.0), 100.0 / (ml + 100.0))


def prob_to_american(p):
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p >= 0.5, -100.0 * p / (1 - p), 100.0 * (1 - p) / p)


def american_to_decimal(ml):
    ml = np.asarray(ml, dtype=float)
    return np.where(ml < 0, 1 + 100.0 / -ml, 1 + ml / 100.0)


def _clean_lines(o: pd.DataFrame) -> pd.DataFrame:
    o = o.copy()
    for c in o.columns:
        if c not in ("provider", "details"):
            o[c] = pd.to_numeric(o[c], errors="coerce")
    for c in [c for c in o.columns if c.endswith("spread") and "odds" not in c]:
        o.loc[o[c].abs() > 70, c] = np.nan
    for c in [c for c in o.columns if c.endswith("total")]:
        o.loc[(o[c] < 90) | (o[c] > 230), c] = np.nan
    for c in [c for c in o.columns if c.endswith("_ml") or c.endswith("_odds")]:
        o.loc[o[c].abs() < 100, c] = np.nan     # american odds can't be in (-100, 100)
    return o


def build_market(odds: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    o = odds[~odds.provider.fillna("").str.contains("|".join(EXCLUDE_PROVIDERS))].copy()
    o = o.merge(games[["game_id", "season"]], on="game_id", how="inner")
    o = _clean_lines(o)
    has_close = o[["close_home_spread", "close_total", "close_home_ml"]].notna().any(axis=1)

    # closing line per book: explicit close if present, else the archived "current" value
    o["c_spread"] = o.close_home_spread.where(has_close, o.spread)
    o["c_total"] = o.close_total.where(has_close, o.total)
    o["c_home_ml"] = o.close_home_ml.where(has_close, o.home_ml)
    o["c_away_ml"] = o.close_away_ml.where(has_close, o.away_ml)
    o["c_home_spread_odds"] = o.close_home_spread_odds.where(has_close, o.home_spread_odds)
    o["c_away_spread_odds"] = o.close_away_spread_odds.where(has_close, o.away_spread_odds)
    o["c_over_odds"] = o.close_over_odds.where(has_close, o.over_odds)
    o["c_under_odds"] = o.close_under_odds.where(has_close, o.under_odds)

    books = o[~o.provider.isin(AGGREGATORS)]
    agg = o[o.provider.isin(AGGREGATORS - {"Opening"})]

    def med(df, cols, suffix):
        # American odds can't be averaged across the +/-100 boundary: aggregate moneylines as probabilities
        df = df.copy()
        ml_cols = [c for c in cols if c.endswith("_ml")]
        for c in ml_cols:
            df[c] = american_to_prob(df[c])
        m = df.groupby("game_id")[cols].median()
        for c in ml_cols:
            m[c] = prob_to_american(m[c])
        m.columns = [f"{c}{suffix}" for c in cols]
        return m

    close_cols = ["c_spread", "c_total", "c_home_ml", "c_away_ml"]
    mk = med(books, close_cols, "_books").join(med(agg, close_cols, "_agg"), how="outer")
    mk["n_books"] = books.groupby("game_id").provider.nunique()
    out = pd.DataFrame(index=mk.index)
    for c in close_cols:
        out[c.replace("c_", "close_")] = mk[f"{c}_books"].fillna(mk[f"{c}_agg"])

    # primary book = the single most common real book in that season (for juice / single-book sims)
    prim = (books.dropna(subset=["c_spread"]).groupby(["season", "provider"]).size()
            .reset_index(name="n").sort_values("n").groupby("season").tail(1)
            .set_index("season").provider.to_dict())
    pb = books[books.provider == books.season.map(prim)].drop_duplicates("game_id").set_index("game_id")
    for c in ["c_home_spread_odds", "c_away_spread_odds", "c_over_odds", "c_under_odds", "c_home_ml", "c_away_ml",
              "c_spread", "c_total"]:
        out[f"pb_{c[2:]}"] = pb[c]

    # opening lines: explicit open fields (2023-24+) or the 'Opening' pseudo-provider (2015-2018)
    op = o[o.open_home_spread.notna() | o.open_total.notna() | o.open_home_ml.notna()]
    op = op[~op.provider.isin(AGGREGATORS)]
    om = med(op, ["open_home_spread", "open_total", "open_home_ml", "open_away_ml"], "")
    opening = med(o[o.provider == "Opening"], ["spread", "total", "home_ml", "away_ml"], "")
    opening.columns = ["open_home_spread", "open_total", "open_home_ml", "open_away_ml"]
    om = om.combine_first(opening)
    out["open_spread"] = om.open_home_spread
    out["open_total"] = om.open_total
    out["open_home_ml"] = om.open_home_ml
    out["open_away_ml"] = om.open_away_ml
    # opening juice from primary book (where available)
    pbo = op[op.provider == op.season.map(prim)].drop_duplicates("game_id").set_index("game_id")
    for c in ["open_home_spread_odds", "open_away_spread_odds", "open_over_odds", "open_under_odds",
              "open_home_spread", "open_total", "open_home_ml", "open_away_ml"]:
        out[f"pb_{c}"] = pbo[c] if c in pbo else np.nan
    out["n_books"] = mk["n_books"]
    out["primary_book"] = out.index.map(games.set_index("game_id").season.map(prim))
    return out.reset_index()


# ------------------------------------------------------------------------------------------
# box score -> possessions & components
# ------------------------------------------------------------------------------------------

def build_games() -> tuple[pd.DataFrame, pd.DataFrame]:
    games = _read("games")
    tb = _read("teambox").drop(columns=["season"], errors="ignore")
    odds = _read("odds").drop(columns=["season"], errors="ignore")

    games = games.drop_duplicates("game_id")
    # D1 membership: teams with >= 12 games in the season's D1 scoreboards
    cnt = pd.concat([games[["season", "home_id"]].rename(columns={"home_id": "tid"}),
                     games[["season", "away_id"]].rename(columns={"away_id": "tid"})]).value_counts()
    d1 = set(cnt[cnt >= 12].index)
    # season in progress: inherit last season's D1 set + any team in a conference that had D1 teams
    for s in sorted(games.season.unique()):
        if not any(ss == s for ss, _ in d1) and any(ss == s - 1 for ss, _ in d1):
            prev_d1 = {t for ss, t in d1 if ss == s - 1}
            gs = games[games.season == s - 1]
            conf_of = dict(zip(gs.home_id, gs.home_conf)) | dict(zip(gs.away_id, gs.away_conf))
            d1_confs = {conf_of.get(t) for t in prev_d1} - {None}
            cur = games[games.season == s]
            cur_conf = dict(zip(cur.home_id, cur.home_conf)) | dict(zip(cur.away_id, cur.away_conf))
            for t, c in cur_conf.items():
                if t in prev_d1 or (c in d1_confs and cnt.get((s, t), 0) >= 1):
                    d1.add((s, t))
    games["home_d1"] = [(s, t) in d1 for s, t in zip(games.season, games.home_id)]
    games["away_d1"] = [(s, t) in d1 for s, t in zip(games.season, games.away_id)]
    games["date"] = (pd.to_datetime(games.date_utc, utc=True)
                     .dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize())

    tb = tb.drop_duplicates(["game_id", "team_id"])
    if "tov" not in tb or tb["tov"].isna().any():
        tb["tov"] = tb.get("tov").fillna(tb.tov_player.fillna(0) + tb.tov_team.fillna(0))
    keep = ["fgm", "fga", "tpm", "tpa", "ftm", "fta", "oreb", "dreb", "treb", "ast", "stl", "blk", "tov", "pf"]
    h = tb[["game_id", "team_id"] + keep].rename(columns={k: f"h_{k}" for k in keep}).rename(columns={"team_id": "home_id"})
    a = tb[["game_id", "team_id"] + keep].rename(columns={k: f"a_{k}" for k in keep}).rename(columns={"team_id": "away_id"})
    g = games.merge(h, on=["game_id", "home_id"], how="left").merge(a, on=["game_id", "away_id"], how="left")

    g["n_periods"] = g.n_periods.where(g.n_periods >= 2, 2)
    g["minutes"] = 40 + 5 * (g.n_periods - 2)
    for s in ("h", "a"):
        g[f"{s}_poss"] = g[f"{s}_fga"] - g[f"{s}_oreb"] + g[f"{s}_tov"] + FT_POSS * g[f"{s}_fta"]
    g["poss"] = (g.h_poss + g.a_poss) / 2
    g["tempo"] = g.poss * 40 / g.minutes
    g["margin"] = g.home_score - g.away_score
    g["total_pts"] = g.home_score + g.away_score
    g["reg_tie"] = False  # filled below (regulation margin unknown without linescores; OT flag used instead)
    g["ot"] = g.n_periods > 2

    mk = build_market(odds, games)
    g = g.merge(mk, on="game_id", how="left")

    ok = (g.completed & g.home_d1 & g.away_d1 & g.h_fga.notna() & g.a_fga.notna()
          & g.home_score.notna() & (g.poss > 40) & (g.poss < 110))
    gd = g[ok].sort_values(["date", "game_id"]).reset_index(drop=True)
    # upcoming / non-final games are kept separately for live prediction
    return gd, g


def team_game_rows(g: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game: offense team vs defense team, with component numerators/denominators."""
    rows = []
    for o, d, sign in (("h", "a", 1), ("a", "h", -1)):
        r = pd.DataFrame({
            "game_id": g.game_id, "season": g.season, "date": g.date,
            "off": g[f"{'home' if o == 'h' else 'away'}_id"], "def": g[f"{'home' if d == 'h' else 'away'}_id"],
            "loc": np.where(g.neutral, 0, sign),
            "poss": g.poss, "tempo": g.tempo, "minutes": g.minutes,
            "pts": g["home_score" if o == "h" else "away_score"],
            "fga": g[f"{o}_fga"], "fgm": g[f"{o}_fgm"], "tpa": g[f"{o}_tpa"], "tpm": g[f"{o}_tpm"],
            "fta": g[f"{o}_fta"], "ftm": g[f"{o}_ftm"], "oreb": g[f"{o}_oreb"], "tov": g[f"{o}_tov"],
            "opp_dreb": g[f"{d}_dreb"],
        })
        rows.append(r)
    tg = pd.concat(rows, ignore_index=True)
    tg["ppp"] = tg.pts / tg.poss
    tg["twa"] = tg.fga - tg.tpa
    tg["twm"] = tg.fgm - tg.tpm
    tg["miss"] = tg.fga - tg.fgm
    return tg


def main():
    os.makedirs(PROC, exist_ok=True)
    gd, gall = build_games()
    gd.to_parquet(os.path.join(PROC, "games.parquet"), index=False)
    gall.to_parquet(os.path.join(PROC, "games_all.parquet"), index=False)
    tg = team_game_rows(gd)
    tg.to_parquet(os.path.join(PROC, "teamgame.parquet"), index=False)
    print("games", gd.shape, "by season:")
    summ = gd.groupby("season").agg(n=("game_id", "size"), close_spread=("close_spread", "count"),
                                    open_spread=("open_spread", "count"), close_total=("close_total", "count"),
                                    open_total=("open_total", "count"), close_ml=("close_home_ml", "count"),
                                    open_ml=("open_home_ml", "count"), tempo=("tempo", "mean"),
                                    ppp=("total_pts", lambda x: np.nan), primary=("primary_book", "first"))
    summ["ppp"] = gd.groupby("season").apply(lambda x: (x.total_pts / (2 * x.poss)).mean(), include_groups=False)
    print(summ.to_string())


if __name__ == "__main__":
    main()
