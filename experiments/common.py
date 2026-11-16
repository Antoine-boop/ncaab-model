"""Shared experiment frame: base + all KEEP features so far."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src")); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import exp as E
from e06_team_hca import G as G6, M1, T1

def build():
    G = G6.merge(pd.read_pickle("experiments/.cache/conf_feats.pkl"), on="game_id", how="left")
    G["e_m"] = G.margin - G.eff_margin; G["e_t"] = G.total_pts - G.eff_total
    long = pd.concat([pd.DataFrame({"game_id": G.game_id, "season": G.season, "date": G.date, "t": G.home_id, "em": G.e_m, "et": G.e_t}),
                      pd.DataFrame({"game_id": G.game_id, "season": G.season, "date": G.date, "t": G.away_id, "em": -G.e_m, "et": G.e_t})]).sort_values(["t", "date", "game_id"])
    grp = long.groupby(["season", "t"])
    long["fm"] = grp.em.transform(lambda x: x.shift().ewm(halflife=6).mean()).fillna(0).to_numpy()
    long["ft"] = grp.et.transform(lambda x: x.shift().ewm(halflife=6).mean()).fillna(0).to_numpy()
    key = long.set_index(["game_id", "t"])
    for s, col in (("h", "home_id"), ("a", "away_id")):
        idx = pd.MultiIndex.from_arrays([G.game_id, G[col]])
        G[f"{s}_fm"] = key.fm.reindex(idx).to_numpy(); G[f"{s}_ft"] = key.ft.reindex(idx).to_numpy()
    G["form_m"] = G.h_fm - G.a_fm; G["form_t"] = G.h_ft + G.a_ft
    return G

M_KEEP = M1 + ["hca_team", "conf_m", "form_m"]
T_KEEP = T1 + ["arena_tot", "conf_t", "form_t"]
