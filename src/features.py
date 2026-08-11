"""
Leakage-free feature pipeline for Sackmann-format ATP match CSVs.

For every match row, attaches each player's serve/return rates computed ONLY
from their PRIOR matches (expanding, cross-season), so backtests are honest.

Output: one row per (match, player) in `long_features`, and one row per match
in `match_features` with w_prior_* / l_prior_* columns plus opponent id/name.
"""

from pathlib import Path
import random

import numpy as np
import pandas as pd

DATA_DIR = Path("./tennis-sackmann-archive/atp")

# Raw serve-count columns present for each side in the Sackmann files
STATS = [
    "ace",
    "df",
    "svpt",
    "1stIn",
    "1stWon",
    "2ndWon",
    "SvGms",
    "bpSaved",
    "bpFaced",
]

# Count columns we accumulate for prior features (own serve + opponent serve)
COUNT_COLS = [s for s in STATS] + [f"opp_{s}" for s in STATS]


def load_seasons(seasons, data_dir=DATA_DIR):
    """Load and concatenate one or more season CSVs, adding a unique match_id."""
    frames = []
    for season in seasons:
        df = pd.read_csv(data_dir / f"atp_matches_{season}.csv")
        df["season"] = season
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["match_id"] = df.index  # unique key to merge features back onto matches
    return df


def build_long(df):
    """One row per (match, player): own + opponent serve counts, ids, names."""

    def side(p, o, id_col, name_col, opp_id_col, opp_name_col):
        cols = {
            "match_id": "match_id",
            "season": "season",
            "tourney_date": "tourney_date",
            "match_num": "match_num",
            id_col: "player_id",
            name_col: "player_name",
            opp_id_col: "opp_id",
            opp_name_col: "opp_name",
        }
        cols.update({f"{p}_{s}": s for s in STATS})
        cols.update({f"{o}_{s}": f"opp_{s}" for s in STATS})
        out = df[list(cols)].rename(columns=cols)
        out["won"] = int(p == "w")
        return out

    long = pd.concat(
        [
            side("w", "l", "winner_id", "winner_name", "loser_id", "loser_name"),
            side("l", "w", "loser_id", "loser_name", "winner_id", "winner_name"),
        ],
        ignore_index=True,
    )
    return long


# By default, constructs features over career and last 50, 25, and 10 matches
def add_prior_features(long, windows=(None, 50, 25, 10)):
    """Leakage-free prior rates over multiple lookback windows.

    windows: iterable of ints (last-N matches) and/or None (expanding, all history).
             Ints produce suffixed columns (e.g. prior_first_won_pct_w50);
             None produces the unsuffixed expanding versions.
    """
    long = long.sort_values(
        ["player_id", "tourney_date", "match_num"], kind="mergesort"
    ).reset_index(drop=True)

    g_ids = long["player_id"]
    filled = long[COUNT_COLS].fillna(0)
    long["prior_matches"] = long.groupby("player_id").cumcount()

    rate_defs = {
        "first_in_pct": lambda c: c["1stIn"] / c["svpt"],
        "first_won_pct": lambda c: c["1stWon"] / c["1stIn"],
        "second_won_pct": lambda c: c["2ndWon"] / (c["svpt"] - c["1stIn"]),
        "first_ret_won_pct": lambda c: (c["opp_1stIn"] - c["opp_1stWon"])
        / c["opp_1stIn"],
        "second_ret_won_pct": lambda c: (
            c["opp_svpt"] - c["opp_1stIn"] - c["opp_2ndWon"]
        )
        / (c["opp_svpt"] - c["opp_1stIn"]),
    }

    for w in windows:
        if w is None:
            counts = filled.groupby(g_ids).cumsum() - filled  # expanding, excl. current
            suffix = ""
        else:
            counts = (
                filled.groupby(g_ids)
                .apply(lambda df: df.shift(1).rolling(w, min_periods=1).sum())
                .reset_index(drop=True)
            )
            suffix = f"_w{w}"
            long[f"prior_matches{suffix}"] = long["prior_matches"].clip(upper=w)

        for name, fn in rate_defs.items():
            long[f"prior_{name}{suffix}"] = fn(counts)

    rate_cols = [c for c in long.columns if c.startswith("prior_") and "pct" in c]
    long[rate_cols] = long[rate_cols].replace([np.inf, -np.inf], np.nan)
    return long


def build_match_features(seasons, data_dir=DATA_DIR):
    """Returns (match_features, long_features).

    match_features: original match rows + w_prior_* and l_prior_* rate columns.
    long_features:  one row per (match, player) with player/opponent id & name,
                    observed counts, and leakage-free prior features.
    """
    matches = load_seasons(seasons, data_dir)
    long = add_prior_features(build_long(matches))
    keep = ["match_id", "player_id"] + [
        c for c in long.columns if c.startswith("prior_")
    ]
    f = long[keep]

    match_features = (
        matches.merge(
            f.rename(columns={c: f"w_{c}" for c in keep if c not in ("match_id",)}),
            left_on=["match_id", "winner_id"],
            right_on=["match_id", "w_player_id"],
            how="left",
        )
        .merge(
            f.rename(columns={c: f"l_{c}" for c in keep if c not in ("match_id",)}),
            left_on=["match_id", "loser_id"],
            right_on=["match_id", "l_player_id"],
            how="left",
        )
        .drop(columns=["w_player_id", "l_player_id"])
    )

    return match_features, long


# Checks features are being shifted and calculated correctly
def sanity_check(n_players=5, window=25, seed=None):
    long = pd.read_csv("./features/long_features.csv")
    rng = random.Random(seed)

    eligible = long.groupby("player_id").size()
    candidates = eligible[eligible > window + 5].index.tolist()
    tests = rng.sample(candidates, n_players)

    for pid in tests:
        p = (
            long[long["player_id"] == pid]
            .sort_values(["tourney_date", "match_num"])
            .reset_index(drop=True)
        )
        name = p["player_name"].iloc[0]

        # 1) counter check: prior_matches should be 0,1,2,...
        assert (p["prior_matches"] == range(len(p))).all(), f"{name}: counter broken"

        # 2) rate check at a random row past the window
        i = rng.randrange(window + 1, len(p))
        prior = p.iloc[i - window : i]  # exactly the last `window` matches
        hand = prior["1stWon"].sum() / prior["1stIn"].sum()
        got = p.iloc[i][f"prior_first_won_pct_w{window}"]
        ok = np.isclose(hand, got, equal_nan=True)
        print(
            f"{name}: hand={hand:.4f} pipeline={got:.4f} {'OK' if ok else 'MISMATCH'}"
        )
        assert ok, f"{name}: window rate mismatch at row {i}"


if __name__ == "__main__":
    seasons = range(2000, 2027)  # adjust to whatever your archive contains
    match_features, long_features = build_match_features(seasons)

    print(
        match_features[
            [
                "tourney_date",
                "winner_name",
                "loser_name",
                "w_prior_matches",
                "w_prior_first_won_pct",
                "l_prior_matches",
                "l_prior_first_ret_won_pct",
            ]
        ].head(10)
    )

    match_features.to_csv("./features/match_features.csv", index=False)
    long_features.to_csv("./features/long_features.csv", index=False)
    print(
        f"\nSaved {len(match_features)} matches, {len(long_features)} player-match rows."
    )
