"""
Schedule / travel / altitude context features (all known before tip-off).

  rest_h, rest_a      days since each team's previous game (capped at 8; season opener = 8)
  b2b_h, b2b_a        played the previous day
  g7_h, g7_a          games in the previous 7 days
  trav_h, trav_a      great-circle miles (thousands) from each team's home state to the venue
  tz_a                time zones crossed by the away/second team
  elev                venue elevation (thousand ft) for known high-altitude venues, else 0
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STATE_LL = {
    "AL": (32.8, -86.8), "AK": (61.4, -150.0), "AZ": (34.2, -111.7), "AR": (34.9, -92.4), "CA": (36.8, -119.4),
    "CO": (39.0, -105.5), "CT": (41.6, -72.7), "DE": (39.0, -75.5), "DC": (38.9, -77.0), "FL": (28.6, -82.4),
    "GA": (32.7, -83.4), "HI": (21.3, -157.8), "ID": (44.1, -114.6), "IL": (40.0, -89.2), "IN": (39.9, -86.3),
    "IA": (42.0, -93.5), "KS": (38.5, -98.4), "KY": (37.5, -85.3), "LA": (31.0, -92.0), "ME": (45.3, -69.2),
    "MD": (39.0, -76.8), "MA": (42.3, -71.8), "MI": (43.3, -84.5), "MN": (46.3, -94.3), "MS": (32.7, -89.7),
    "MO": (38.4, -92.5), "MT": (47.0, -109.6), "NE": (41.5, -99.8), "NV": (38.5, -117.0), "NH": (43.7, -71.6),
    "NJ": (40.2, -74.6), "NM": (34.4, -106.1), "NY": (42.9, -75.5), "NC": (35.5, -79.4), "ND": (47.5, -100.5),
    "OH": (40.3, -82.8), "OK": (35.6, -97.5), "OR": (43.9, -120.6), "PA": (40.9, -77.8), "RI": (41.7, -71.5),
    "SC": (33.9, -80.9), "SD": (44.4, -100.2), "TN": (35.9, -86.4), "TX": (31.5, -99.3), "UT": (39.3, -111.7),
    "VT": (44.0, -72.7), "VA": (37.5, -78.9), "WA": (47.4, -120.5), "WV": (38.6, -80.6), "WI": (44.6, -89.9),
    "WY": (43.0, -107.6), "Bahamas": (25.0, -77.4), "Virgin Islands": (18.3, -64.9), "Mexico": (21.2, -86.8),
    "Cayman Islands": (19.3, -81.4), "Puerto Rico": (18.2, -66.5), "Jamaica": (18.1, -77.3), "PQ": (45.5, -73.6),
    "Northern Ireland": (54.6, -5.9), "NB": (46.5, -66.5), "ON": (43.7, -79.4), "BC": (49.3, -123.1),
}

# venue city -> elevation (ft) for the high-altitude D1 venues
HIGH_ALT = {
    "Colorado Springs": 6600, "USAF Academy": 7200, "Air Force Academy": 7200, "U.S. Air Force Academy": 7200,
    "Laramie": 7165, "Flagstaff": 6900, "Cedar City": 5846, "Boulder": 5328, "Albuquerque": 5312, "Denver": 5280,
    "Fort Collins": 5003, "Bozeman": 4820, "Orem": 4774, "Greeley": 4658, "Provo": 4551, "Logan": 4535,
    "Reno": 4505, "Pocatello": 4462, "Ogden": 4300, "Salt Lake City": 4226, "Las Cruces": 3900, "El Paso": 3740,
    "Lubbock": 3256, "Missoula": 3209,
}


def _miles(a, b):
    la1, lo1 = np.radians(a[..., 0]), np.radians(a[..., 1])
    la2, lo2 = np.radians(b[..., 0]), np.radians(b[..., 1])
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 3959 * 2 * np.arcsin(np.sqrt(h))


def add_context(G: pd.DataFrame, games_all: pd.DataFrame) -> pd.DataFrame:
    G = G.copy()
    cols = ["game_id", "season", "date", "home_id", "away_id", "neutral", "venue_state"]
    ga = pd.concat([games_all.loc[games_all.completed, cols], G[cols]]).drop_duplicates("game_id")
    # ---- rest / density ----
    long = pd.concat([ga[["season", "date", "home_id", "game_id"]].rename(columns={"home_id": "t"}),
                      ga[["season", "date", "away_id", "game_id"]].rename(columns={"away_id": "t"})])
    long = long.drop_duplicates(["t", "game_id"]).sort_values(["t", "date", "game_id"])
    long["prev"] = long.groupby(["season", "t"]).date.shift()
    long["rest"] = (long.date - long.prev).dt.days.fillna(8).clip(0, 8)
    # games in previous 7 days
    long["g7"] = 0
    for k in range(1, 6):
        prevk = long.groupby(["season", "t"]).date.shift(k)
        long["g7"] += ((long.date - prevk).dt.days <= 7).astype(int)
    key = long.set_index(["game_id", "t"])
    for s, col in (("h", "home_id"), ("a", "away_id")):
        idx = pd.MultiIndex.from_arrays([G.game_id, G[col]])
        G[f"rest_{s}"] = key.rest.reindex(idx).fillna(8).to_numpy()
        G[f"g7_{s}"] = key.g7.reindex(idx).fillna(0).to_numpy()
        G[f"b2b_{s}"] = (G[f"rest_{s}"] <= 1).astype(float)
    # ---- travel ----
    homes = ga[~ga.neutral & ga.venue_state.notna()]
    home_state = homes.groupby("home_id").venue_state.agg(lambda x: x.value_counts().index[0])
    ll = lambda states: np.array([STATE_LL.get(s, (np.nan, np.nan)) for s in states], dtype=float)
    v = ll(G.venue_state.fillna(""))
    hs = ll(G.home_id.map(home_state).fillna(""))
    as_ = ll(G.away_id.map(home_state).fillna(""))
    G["trav_h"] = np.nan_to_num(_miles(hs, v) / 1000, nan=0.0)
    G["trav_a"] = np.nan_to_num(_miles(as_, v) / 1000, nan=0.0)
    G.loc[~G.neutral, "trav_h"] = 0.0
    G["tz_a"] = np.nan_to_num(np.abs(np.round(v[:, 1] / 15) - np.round(as_[:, 1] / 15)), nan=0.0)
    # ---- altitude ----
    G["elev"] = G.venue_city.map(HIGH_ALT).fillna(0).to_numpy() / 1000
    # derived
    G["rest_diff"] = G.rest_h.clip(1, 4) - G.rest_a.clip(1, 4)
    G["b2b_diff"] = G.b2b_h - G.b2b_a
    G["g7_diff"] = G.g7_h - G.g7_a
    G["trav_diff"] = G.trav_a - G.trav_h
    G["elev_home"] = G.elev * (~G.neutral).astype(float)
    G["rest_sum"] = G.rest_h.clip(1, 4) + G.rest_a.clip(1, 4)
    G["b2b_any"] = np.maximum(G.b2b_h, G.b2b_a)
    return G
