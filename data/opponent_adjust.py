# opponent-adjusted EPA/play and yards/play, added as separate features
"""Opponent-adjusted efficiency stats (Section 1 of the spec). Raw rolling
stats (data/team_stats.py) treat a good game against a bad defense
identically to the same production against a good defense. This normalizes
each game's per-play production by the opponent's own pre-game rolling
strength before rolling it back up, so the season-long averages reflect
quality of opposition, not just raw counting stats.

Single-pass adjustment -- the spec's "or simpler" alternative to full
iterative DVOA. Each game's adjustment uses the opponent's own multi-game
rolling average coming into that game, not a from-scratch iterative solve.
Since the opponent's rolling average is already an aggregate over several
games (not one noisy game), most of the benefit of iterating further is
already captured without the complexity -- and risk of leakage bugs -- of
recursively re-deriving ratings from each other.

Both raw and adjusted columns are kept as separate model features
(model/train.py's STAT_COLS) rather than replacing the raw ones -- they're
correlated but not redundant, and a regularized model can weigh each on
its own merits.
"""

import pandas as pd

# offense_col -> the opponent stat (their defense) it should be adjusted
# against, and vice versa.
OPPONENT_STAT = {
    "off_epa_per_play": "def_epa_per_play", "def_epa_per_play": "off_epa_per_play",
    "off_ypp": "def_ypp", "def_ypp": "off_ypp",
}
ADJ_COLS = [f"{col}_adj" for col in OPPONENT_STAT]


def add_opponent_adjusted_columns(team_game_stats: pd.DataFrame, rolling: pd.DataFrame) -> pd.DataFrame:
    """Adds a `{col}_adj` per-game column to team_game_stats for each key in
    OPPONENT_STAT. `rolling` must be the RAW rolling table (already
    pre-game-shifted -- see data/team_stats.py's _rolling_for_team_season)
    so the opponent's incoming strength reflects only what was knowable
    before this specific game was played, same no-leakage guarantee as the
    raw rolling stats."""
    rolling_cols = [f"{c}_avg" for c in OPPONENT_STAT]

    opp_rolling = rolling[["season", "week", "team"] + rolling_cols].rename(
        columns={"team": "opponent", **{f"{c}_avg": f"opp_{c}_avg" for c in OPPONENT_STAT}}
    )
    league_avg = (
        rolling.groupby(["season", "week"])[rolling_cols].mean()
        .rename(columns={f"{c}_avg": f"league_{c}_avg" for c in OPPONENT_STAT})
        .reset_index()
    )

    games = team_game_stats.merge(opp_rolling, on=["season", "week", "opponent"], how="left")
    games = games.merge(league_avg, on=["season", "week"], how="left")

    for col, opp_col in OPPONENT_STAT.items():
        # raw production, minus how strong the opponent's own side normally
        # is, plus the league-average version of that same opponent stat --
        # nets out to "what this performance would look like against an
        # average opponent." NaN (opponent's own first game of the season,
        # nothing to adjust against yet) propagates rather than guessing.
        games[f"{col}_adj"] = games[col] - games[f"opp_{opp_col}_avg"] + games[f"league_{opp_col}_avg"]

    return games


def _roll_adjusted(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("week").copy()
    for col in ADJ_COLS:
        g[f"{col}_avg"] = g[col].expanding().mean().shift(1)
    return g


def roll_opponent_adjusted(games_with_adj: pd.DataFrame) -> pd.DataFrame:
    """Rolls the per-game *_adj columns into *_adj_avg pre-game rolling
    averages -- same expanding-mean-shifted-by-1 pattern as
    data/team_stats.py's raw rolling stats, so it carries the same
    no-leakage guarantee."""
    rolled = (
        games_with_adj.groupby(["team", "season"])
        .apply(_roll_adjusted)
        .reset_index(level=["team", "season"])
    )
    keep = ["season", "week", "team"] + [f"{c}_avg" for c in ADJ_COLS]
    return rolled[keep].sort_values(["team", "season", "week"]).reset_index(drop=True)
