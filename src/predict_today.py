"""
Daily picks: refresh current-season data, project today's (or a given date's) D1 games and price
spreads / totals / moneylines against DraftKings lines archived by ESPN (open + current).

    python src/predict_today.py                 # today (America/New_York)
    python src/predict_today.py --date 2026-11-17

Writes reports/picks_<date>.csv and prints the slate.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest as BT  # noqa: E402
import model as MD  # noqa: E402
import scrape_espn as SE  # noqa: E402
from build_dataset import american_to_decimal, american_to_prob  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
REPORTS = os.path.join(ROOT, "reports")


LINE_COLS = ["provider", "spread", "total", "home_ml", "away_ml", "home_spread_odds", "away_spread_odds",
             "over_odds", "under_odds", "open_home_spread", "open_total", "open_home_ml", "open_away_ml",
             "open_home_spread_odds", "open_away_spread_odds", "open_over_odds", "open_under_odds"]


def current_season(d: dt.date) -> int:
    return d.year + 1 if d.month >= 8 else d.year


def fetch_slate(d: dt.date) -> pd.DataFrame:
    sb = SE.get_json(SE.SB_URL.format(d=d.strftime("%Y%m%d")))
    rows = SE.parse_scoreboard(sb or {}, current_season(d))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[~df.completed & df.status.isin(["STATUS_SCHEDULED", "STATUS_DELAYED"])].copy()
    odds = []
    for gid in df.game_id:
        odds.extend(SE.parse_odds(SE.get_json(SE.ODDS_URL.format(e=gid)), gid))
    o = pd.DataFrame(odds)
    if not o.empty:
        o = o[~o.provider.fillna("").str.contains("Live")]
        o = o.sort_values("provider_id").drop_duplicates("game_id")
        df = df.merge(o, on="game_id", how="left")
    for c in LINE_COLS:                     # lines may not be posted yet
        if c not in df.columns:
            df[c] = np.nan
    df["date"] = pd.Timestamp(d)
    return df


MISMATCH = 20.0                                # buy games: model under-projects big favourites
MIN_EDGE = {"spread": 1.5, "total": 2.0}       # model must disagree with the number, not just the price
NOV_SPREAD_MIN_EV = 0.06                       # November spreads: weakest month in the backtest (+2.5% at EV>=3%)


def kelly_fraction(p, dec, frac=0.25, cap=0.01):
    b = dec - 1
    f = (p * b - (1 - p)) / b
    return np.clip(f * frac, 0, cap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    ap.add_argument("--no-refresh", action="store_true")
    ap.add_argument("--min-ev", type=float, default=0.03)
    ap.add_argument("--allow-mismatch", action="store_true")
    args = ap.parse_args()
    today = dt.date.fromisoformat(args.date) if args.date else pd.Timestamp.now(tz="America/New_York").date()
    season = current_season(today)

    if not args.no_refresh:
        print(f"refreshing season {season} through {today - dt.timedelta(days=1)} ...", flush=True)
        subprocess.run([sys.executable, os.path.join(ROOT, "src", "scrape_espn.py"), str(season), "--refresh",
                        "--end", str(today - dt.timedelta(days=1))], check=True)
        subprocess.run([sys.executable, os.path.join(ROOT, "src", "build_dataset.py")], check=True,
                       stdout=subprocess.DEVNULL)

    games = pd.read_parquet(os.path.join(PROC, "games.parquet"))
    games = games[games.date < pd.Timestamp(today)]
    slate = fetch_slate(today)
    if slate.empty:
        print("no scheduled games found")
        return
    d1 = set(games[games.season == season].home_id) | set(games[games.season == season].away_id)
    d1 |= set(games[games.season == season - 1].home_id) | set(games[games.season == season - 1].away_id)
    slate = slate[slate.home_id.isin(d1) & slate.away_id.isin(d1)].copy()
    print(f"{len(slate)} D1-vs-D1 games on {today}", flush=True)

    up = slate[["game_id", "season", "date", "season_type", "neutral", "conf_game", "home_id", "away_id",
                "home_name", "away_name", "n_periods", "home_conf", "away_conf", "venue_city", "venue_state"]].copy()
    up["n_periods"] = 2
    cache = os.path.join(PROC, f"upcoming_{today}.parquet")
    s2 = pd.read_pickle(os.path.join(PROC, "stage2.pkl"))
    if args.no_refresh and os.path.exists(cache):
        U = pd.read_parquet(cache)
    else:
        games_all = pd.read_parquet(os.path.join(PROC, "games_all.parquet"))
        games_all = games_all[games_all.date < pd.Timestamp(today)]
        G, aux = MD.run_pipeline(games, upcoming=up, verbose=False, games_all=games_all)
        G = MD.apply_stage2(G, s2)
        U = G[G.game_id.isin(up.game_id)].copy()
        U.to_parquet(cache, index=False)
    line_cols = LINE_COLS + ["date_utc"]
    U = U.drop(columns=[c for c in line_cols if c in U.columns])
    P = U.merge(slate[["game_id"] + line_cols], on="game_id", how="left")
    if P.open_home_spread.isna().all() and P.open_total.isna().all():
        proj = P[["date_utc", "away_name", "home_name", "pred_margin", "pred_total", "p_home"]].round(3)
        out = os.path.join(REPORTS, f"projections_{today}.csv")
        proj.to_csv(out, index=False)
        print(f"no lines posted yet for {today}; projections only -> {out}")
        print(proj.to_string(index=False))
        return

    with open(os.path.join(REPORTS, "calibration.json")) as fh:
        cal = json.load(fh)["open"]       # edges are validated against OPENING lines only
    rows = []
    for _, g in P.iterrows():
        base = dict(tip_utc=g.date_utc, game=f"{g.away_name} @ {g.home_name}" + (" (N)" if g.neutral else ""),
                    proj_home_margin=round(g.pred_margin, 1), proj_total=round(g.pred_total, 1))
        sp_open, tot_open = g.open_home_spread, g.open_total
        mismatch = pd.notna(sp_open) and abs(sp_open) >= MISMATCH
        if pd.notna(sp_open):
            ph = 1 / (1 + np.exp(-(cal["spread"][0] + cal["spread"][1] * (g.pred_margin + sp_open)
                                   / BT.sig_margin(sp_open))))
            for side, p, price, cur_line, cur_price, name in (
                    (1, ph, g.open_home_spread_odds, g.spread, g.home_spread_odds, g.home_name),
                    (-1, 1 - ph, g.open_away_spread_odds, -g.spread if pd.notna(g.spread) else np.nan,
                     g.away_spread_odds, g.away_name)):
                price = price if pd.notna(price) and abs(price) >= 100 else -110.0
                dec = float(american_to_decimal(price))
                line = sp_open * side
                edge = (g.pred_margin + sp_open) * side
                rows.append(base | dict(market="spread", pick=f"{name} {line:+.1f}", open_price=price,
                                        edge_pts=round(edge, 1), p_win=round(p, 3), ev=round(p * (dec - 1) - (1 - p), 3),
                                        now=(f"{cur_line:+.1f} ({cur_price:+.0f})" if pd.notna(cur_line) else "n/a"),
                                        available_now=bool(pd.notna(cur_line) and cur_line >= line),
                                        mismatch=mismatch, min_edge=edge >= MIN_EDGE["spread"],
                                        stake_pct=round(float(kelly_fraction(p, dec)) * 100, 2)))
        if pd.notna(tot_open):
            po = 1 / (1 + np.exp(-(cal["total"][0] + cal["total"][1] * (g.pred_total - tot_open)
                                   / BT.sig_total(tot_open))))
            for side, p, price, cur_price, name in ((1, po, g.open_over_odds, g.over_odds, "Over"),
                                                    (-1, 1 - po, g.open_under_odds, g.under_odds, "Under")):
                price = price if pd.notna(price) and abs(price) >= 100 else -110.0
                dec = float(american_to_decimal(price))
                edge = (g.pred_total - tot_open) * side
                avail = pd.notna(g.total) and ((g.total <= tot_open) if side == 1 else (g.total >= tot_open))
                rows.append(base | dict(market="total", pick=f"{name} {tot_open:.1f}", open_price=price,
                                        edge_pts=round(edge, 1), p_win=round(p, 3), ev=round(p * (dec - 1) - (1 - p), 3),
                                        now=(f"{g.total:.1f} ({cur_price:+.0f})" if pd.notna(g.total) else "n/a"),
                                        available_now=bool(avail), mismatch=mismatch,
                                        min_edge=edge >= MIN_EDGE["total"],
                                        stake_pct=round(float(kelly_fraction(p, dec)) * 100, 2)))
    picks = pd.DataFrame(rows)
    os.makedirs(REPORTS, exist_ok=True)
    out = os.path.join(REPORTS, f"picks_{today}.csv")
    picks.to_csv(out, index=False)
    min_ev = np.where((picks.market == "spread") & (today.month == 11), max(args.min_ev, NOV_SPREAD_MIN_EV), args.min_ev)
    q = picks[(picks.ev >= min_ev) & picks.min_edge]
    bets = q[~q.mismatch | args.allow_mismatch].sort_values("ev", ascending=False)
    skipped = q[q.mismatch & ~args.allow_mismatch]
    pd.set_option("display.width", 250)
    cols = ["game", "market", "pick", "open_price", "proj_home_margin", "proj_total", "edge_pts", "p_win", "ev",
            "stake_pct", "now", "available_now"]
    print(f"\nall projections -> {out}")
    print(f"\nBETS (EV >= {args.min_ev:.0%} vs the OPENING number; take only if available_now):")
    print(bets[cols].to_string(index=False) if len(bets) else "  none")
    if len(skipped):
        print(f"\nskipped {len(skipped)} qualifying bets in mismatches (|spread| >= {MISMATCH}); "
              "use --allow-mismatch to include")
    print("\nNote: no moneyline bets - the backtest showed no moneyline edge (see reports/REPORT.md).")


if __name__ == "__main__":
    main()
