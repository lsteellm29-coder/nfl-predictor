# opponent-adjusted EPA/play and yards/play, added as separate features
"""Opponent-adjusted efficiency stats (Section 1 of the spec). Raw rolling
stats (data/team_stats.py) treat a good game against a bad defense
identically to the same production against a good defense. This normalizes
each game's per-play production by the opponent's own pre-game rolling
strength before rolling it back up, so the season-long averages reflect
quality of opposition, not just raw counting stats.

Iterative (DVOA-style) adjustment (Audit Fix Plan Step 4; previously
single-pass). Each pass recomputes "how strong is this opponent" using the
PREVIOUS pass's own adjusted rolling numbers instead of the raw ones, so a
team's rating accounts for the quality of opponents its opponents played
too, not just one level of adjustment -- iterative_opponent_adjust() below
runs 3 passes by default. Each pass still only ever uses a pre-game
(already shifted) rolling average as the opponent reference, same
no-leakage guarantee as the raw rolling stats and as the original
single-pass version -- iterating doesn't relax that, it just refines what
"opponent strength" means before applying the same subtract-and-recenter
formula.

Damped (not naive successive substitution): each pass's raw update is
blended DAMPING_FACTOR-weighted with the running estimate rather than
replacing it outright. This isn't optional polish -- an undamped version
was tested first and empirically diverges. Subtracting an opponent's
already-adjusted (and thus already more spread-out) rating pushes the next
pass's adjustment further from center every round with nothing pulling it
back: measured on the real cached seasons, undamped std dev of
off_epa_per_play_adj_avg grew pass over pass (0.140 -> 0.151 -> 0.177 ->
0.187 -> 0.202 across 5 passes, versus 0.119 for the raw unadjusted stat)
instead of settling down. Damping at 0.3 converges cleanly and
monotonically instead (0.140 -> 0.123 -> 0.121 -> 0.119 -> 0.118 -- each
pass's shift shrinking geometrically), which is what "iterative" is
supposed to look like.

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

# How much of each iterative pass's freshly-recomputed value to accept vs.
# how much of the running estimate to keep -- see this module's docstring
# for why an undamped (1.0) version diverges. Not tuned finely; 0.3 was
# the first value tried that converged cleanly and there was no sign
# convergence was sensitive to the exact number (0.5 also stabilized, just
# with a small residual oscillation instead of a clean monotonic settle).
DAMPING_FACTOR = 0.3


def add_opponent_adjusted_columns(team_game_stats: pd.DataFrame, rolling: pd.DataFrame,
                                   rolling_col_suffix: str = "_avg") -> pd.DataFrame:
    """Adds a `{col}_adj` per-game column to team_game_stats for each key in
    OPPONENT_STAT. `rolling` must already be pre-game-shifted (see
    data/team_stats.py's _rolling_for_team_season) so the opponent's
    incoming strength reflects only what was knowable before this specific
    game was played, same no-leakage guarantee as the raw rolling stats.

    `rolling` can be the raw rolling table (`rolling_col_suffix="_avg"`,
    the default -- first pass) or a prior pass's own adjusted rolling table
    from roll_opponent_adjusted() (`rolling_col_suffix="_adj_avg"` -- a
    later iterative pass); `rolling_col_suffix` just names which column
    naming convention `rolling` uses for each of OPPONENT_STAT's keys, so
    the same merge-and-subtract logic works for either input."""
    rolling_cols = [f"{c}{rolling_col_suffix}" for c in OPPONENT_STAT]

    opp_rolling = rolling[["season", "week", "team"] + rolling_cols].rename(
        columns={"team": "opponent", **{f"{c}{rolling_col_suffix}": f"opp_{c}_avg" for c in OPPONENT_STAT}}
    )
    league_avg = (
        rolling.groupby(["season", "week"])[rolling_cols].mean()
        .rename(columns={f"{c}{rolling_col_suffix}": f"league_{c}_avg" for c in OPPONENT_STAT})
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


def _blend(new: pd.DataFrame, prev: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """alpha*new + (1-alpha)*prev per OPPONENT_STAT's *_adj_avg column,
    joined on (season, week, team). Both frames are already in *_adj_avg
    column-naming space (this is only ever called from pass 2 onward)."""
    cols = [f"{c}_adj_avg" for c in OPPONENT_STAT]
    merged = new.merge(prev, on=["season", "week", "team"], suffixes=("", "_prev"))
    blended = new.copy()
    for col in cols:
        blended[col] = alpha * merged[col] + (1 - alpha) * merged[f"{col}_prev"]
    return blended


def iterative_opponent_adjust(team_game_stats: pd.DataFrame, raw_rolling: pd.DataFrame,
                               n_passes: int = 3, damping: float = DAMPING_FACTOR) -> pd.DataFrame:
    """Runs add_opponent_adjusted_columns + roll_opponent_adjusted `n_passes`
    times, each pass using the previous pass's own (damped) *_adj_avg
    rolling table as the opponent-strength reference instead of the raw
    one -- see this module's docstring for why damping is load-bearing,
    not optional. n_passes=1 reproduces the original single-pass behavior
    exactly (same formula, same input, damping never applies with only one
    pass to blend against). Returns the final pass's rolling table, shaped
    identically to a single-pass call's output (season, week, team,
    {col}_adj_avg per OPPONENT_STAT key) -- callers don't need to know how
    many passes ran."""
    opponent_rolling, suffix = raw_rolling, "_avg"
    blended = None
    for _ in range(n_passes):
        games_adj = add_opponent_adjusted_columns(team_game_stats, opponent_rolling, suffix)
        adj_rolling = roll_opponent_adjusted(games_adj)
        blended = adj_rolling if blended is None else _blend(adj_rolling, blended, damping)
        opponent_rolling, suffix = blended, "_adj_avg"
    return blended
