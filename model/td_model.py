# anytime-TD probability: red-zone/goal-line touch share -> Poisson rate -> P(>=1 TD)
"""Core projection engine for the anytime-TD-only props layer (Anytime-TD-
Only Props Refocus spec). Replaces the yardage/reception projection logic
that used to live in model/player_stats.py -- this is the sole remaining
prop type.

Model shape: a player's expected TD count for a game is their team's
expected red-zone trips that game, times the team's red-zone TD
conversion rate (adjusted for the specific opponent's red-zone defense),
times the player's own share of the team's red-zone touches (rush
attempts + targets inside the 20) -- i.e. "how many scoring chances does
this team get, and how big a slice of them does this player get." This
mirrors how real sports-betting TD models work: team scoring opportunity
x player's share of it, not a flat season-long TD rate that conflates
opportunity with finishing variance.

A QB's own rushing TDs ("goal-line vulture" scores -- sneaks/keepers that
take a TD away from the RB who did the actual work moving the ball) need
no separate mechanism: a QB shows up in this same red-zone rush-touch
table like any other rusher (model/player_stats.py's rb_rushing_game_log
already tracks every real rusher, not just RBs), so their own goal-line
rushing share is captured the same way everyone else's is.
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson

from model.td_ensemble import blend_with_classifier

RED_ZONE_YARDLINE = 20
GOAL_LINE_YARDLINE = 5

# Recency weighting for a player's red-zone share: the last RECENCY_WINDOW
# games each count RECENCY_WEIGHT_MULTIPLIER times as much as an older
# game -- usage changes fast (new OC, an injury opening up touches, a
# rookie earning trust) and a flat season-long average lags behind that.
# Empirically swept (window 3-8, multiplier 1.0-3.0) against the walk-
# forward backtest -- these original judgment-call values already sit
# right at the empirical optimum (Brier 0.1689/AUC 0.6681 vs. 0.1696/
# 0.6656 with recency weighting disabled entirely), no change needed.
RECENCY_WINDOW = 4
RECENCY_WEIGHT_MULTIPLIER = 2.0

# Shrinkage strength (in "worth this many touches of prior") for
# regressing a low-sample player's red-zone share toward the positional
# baseline -- a player with 1 red-zone touch that happened to score isn't
# showing a real 100% red-zone conversion rate, that's noise. Higher =
# more regression for the same sample size.
#
# Empirically swept 4-150 against the walk-forward backtest: Brier and
# AUC improved monotonically from the original guessed values (10/8/8/8)
# all the way to ~75, then plateaued flat through 150 -- 75 sits right at
# the start of that plateau. Per-position values (QB/RB/WR/TE tested
# separately) didn't beat one uniform value, so this stays a single
# number rather than added complexity that doesn't earn its keep.
SHRINKAGE_TOUCHES = {"QB": 75, "RB": 75, "WR": 75, "TE": 75}


def player_red_zone_touches(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per (season, week, team, player_id) red-zone (yardline_100<=20) and
    goal-line (<=5) touches -- rush attempts plus pass targets, whoever
    they were to -- and TDs scored specifically on those touches. This is
    the opportunity signal the whole model is built on: a player's SHARE
    of a team's red-zone touches predicts scoring far better than their
    raw season TD total, which conflates real opportunity with finishing
    luck (a player can score on a 1-in-5 red-zone touch rate one season
    and a 1-in-15 rate the next with the exact same usage)."""
    rush = pbp[(pbp["play_type"] == "run") & pbp["rusher_player_id"].notna()].copy()
    rush["player_id"] = rush["rusher_player_id"]
    rush["is_td"] = rush["rush_touchdown"].fillna(0)

    recv = pbp[(pbp["play_type"] == "pass") & pbp["receiver_player_id"].notna()].copy()
    recv["player_id"] = recv["receiver_player_id"]
    recv["is_td"] = recv["pass_touchdown"].fillna(0)

    cols = ["season", "week", "posteam", "player_id", "yardline_100", "is_td"]
    touches = pd.concat([rush[cols], recv[cols]], ignore_index=True).rename(columns={"posteam": "team"})
    touches = touches[touches["yardline_100"] <= RED_ZONE_YARDLINE].copy()
    touches["is_goal_line"] = touches["yardline_100"] <= GOAL_LINE_YARDLINE

    return touches.groupby(["season", "week", "team", "player_id"]).agg(
        rz_touches=("yardline_100", "size"), rz_tds=("is_td", "sum"), gl_touches=("is_goal_line", "sum"),
    ).reset_index()


def team_red_zone_offense(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per (season, week, team) red-zone trips and TDs on those trips --
    each team's own scoring-opportunity rate and finishing rate, drive-
    based (same fixed_drive/drive_inside20 logic data/team_stats.py's
    red_zone_td_pct already uses, computed fresh here rather than
    imported so this module stays a self-contained projection layer, same
    principle as the rest of model/player_stats.py)."""
    drives = pbp.groupby(["game_id", "season", "week", "posteam", "fixed_drive"]).agg(
        reached_rz=("drive_inside20", "max"), result=("fixed_drive_result", "first"),
    ).reset_index()
    rz = drives[drives["reached_rz"] == 1]
    return rz.groupby(["season", "week", "posteam"]).agg(
        rz_trips=("result", "size"), rz_tds=("result", lambda s: (s == "Touchdown").sum()),
    ).reset_index().rename(columns={"posteam": "team"})


def team_red_zone_defense(pbp: pd.DataFrame) -> pd.DataFrame:
    """Same as team_red_zone_offense, from the defense's side (defteam) --
    how often this team's defense lets an opponent's red-zone trip end in
    a TD. The defensive analog didn't exist anywhere in this codebase
    before (data/team_stats.py only ever computed the offensive side)."""
    drives = pbp.groupby(["game_id", "season", "week", "defteam", "fixed_drive"]).agg(
        reached_rz=("drive_inside20", "max"), result=("fixed_drive_result", "first"),
    ).reset_index()
    rz = drives[drives["reached_rz"] == 1]
    return rz.groupby(["season", "week", "defteam"]).agg(
        rz_trips_allowed=("result", "size"), rz_tds_allowed=("result", lambda s: (s == "Touchdown").sum()),
    ).reset_index().rename(columns={"defteam": "team"})


# NOT implemented: red-zone pass defense split by receiver position (WR/
# TE-specific, the free proxy for 1-on-1 matchup quality this codebase's
# data can actually support -- true player-vs-player coverage data is a
# paid-tier/PFF-level feature). Built and A/B tested against the walk-
# forward backtest: WR/TE Brier went from 0.1681 to 0.1689 and AUC from
# 0.6242 to 0.6222 with the position-specific factor added on top of the
# existing team-wide red-zone defense factor -- both got marginally
# *worse*, not better. Even with a >=20-red-zone-target minimum sample,
# a defense's targets allowed to one specific position inside the 20 is a
# genuinely thin slice of a season (a team might see under 20 red-zone TE
# targets against them all year), thin enough that slicing by position
# added estimation noise instead of real signal. Reverted rather than
# shipped on a negative result.


def red_zone_defense_rank(team_rz_defense: pd.DataFrame, team: str) -> int | None:
    """1-indexed league rank of how often `team`'s defense allows a
    red-zone trip to end in a TD (1 = allows the most) -- matches how
    report/cards.py's reasoning_sentence already phrases "the Nth-most"
    citations elsewhere in this codebase. `team_rz_defense` is expected
    already season-to-date aggregated (model/td_model.py's
    season_to_date output, with a "games" column) -- None if the sample
    is too thin league-wide to rank honestly, same MIN-teams bar
    data/positional_matchups.py's defense_rank() already uses."""
    rated = team_rz_defense[team_rz_defense["rz_trips_allowed"] > 0].copy()
    if team not in set(rated["team"]) or len(rated) < 10:
        return None
    rated["rate"] = rated["rz_tds_allowed"] / rated["rz_trips_allowed"]
    team_rate = rated.loc[rated["team"] == team, "rate"].iloc[0]
    return int((rated["rate"] > team_rate).sum()) + 1


def season_to_date(game_log: pd.DataFrame, group_cols: list, value_cols: list, upto_week: int) -> pd.DataFrame:
    """game_log filtered to weeks < upto_week, summed per group_cols --
    the plain (non-recency-weighted) season-to-date total, used for team-
    level rates where a single flat rate is the right granularity (a
    team's red-zone trip rate doesn't need per-game recency weighting the
    way an individual player's role does)."""
    prior = game_log[game_log["week"] < upto_week]
    if prior.empty:
        return pd.DataFrame(columns=group_cols + value_cols + ["games"])
    games = prior.groupby(group_cols).size().rename("games")
    totals = prior.groupby(group_cols)[value_cols].sum()
    return totals.join(games).reset_index()


def recency_weighted_touch_share(touch_log: pd.DataFrame, upto_week: int,
                                  window: int | None = None, multiplier: float | None = None) -> pd.DataFrame:
    """Per (team, player_id) red-zone touch share among that team's own
    red-zone touches, using games strictly before `upto_week` -- the last
    `window` of those games count `multiplier`x as much as anything
    older. Returns raw weighted touches/tds per player alongside each
    team's weighted touch total, so the caller can compute both the
    player's share (touches / team total) and their own weighted
    conversion rate (tds / touches) from the same weighting.

    window/multiplier default to None and get resolved to the current
    module-level RECENCY_WINDOW/RECENCY_WEIGHT_MULTIPLIER *inside* the
    function body, not as literal parameter defaults -- Python binds a
    `param=MODULE_CONSTANT` default once, at function-definition time, so
    a `td_model.RECENCY_WINDOW = X` reassignment done for a sweep/test
    afterward would silently never take effect with the old signature.
    Caught this exact bug empirically: a parameter sweep across several
    window/multiplier values kept returning byte-for-byte identical
    Brier/AUC no matter what was set, including a supposedly-disabling
    multiplier=1.0 -- the tell that something was stale, not that
    recency weighting genuinely didn't matter."""
    window = RECENCY_WINDOW if window is None else window
    multiplier = RECENCY_WEIGHT_MULTIPLIER if multiplier is None else multiplier
    prior = touch_log[touch_log["week"] < upto_week].copy()
    if prior.empty:
        return pd.DataFrame(columns=["team", "player_id", "w_rz_touches", "w_rz_tds", "w_gl_touches", "n_games"])

    prior = prior.sort_values("week")
    recent_weeks = prior.groupby("team")["week"].apply(lambda s: set(s.drop_duplicates().tail(window)))
    prior["weight"] = prior.apply(
        lambda r: multiplier if r["week"] in recent_weeks.get(r["team"], set()) else 1.0, axis=1)

    prior["w_rz_touches"] = prior["rz_touches"] * prior["weight"]
    prior["w_rz_tds"] = prior["rz_tds"] * prior["weight"]
    prior["w_gl_touches"] = prior["gl_touches"] * prior["weight"]

    out = prior.groupby(["team", "player_id"]).agg(
        w_rz_touches=("w_rz_touches", "sum"), w_rz_tds=("w_rz_tds", "sum"),
        w_gl_touches=("w_gl_touches", "sum"), n_games=("week", "nunique"),
    ).reset_index()
    return out


def team_red_zone_touches_per_game(touch_log: pd.DataFrame, upto_week: int) -> dict:
    """team -> average red-zone TOUCHES (individual plays: rush attempts
    + targets inside the 20) per game, season-to-date through strictly
    earlier weeks -- deliberately unweighted (a team's own red-zone
    opportunity generation is far more stable game-to-game than an
    individual player's role, so this doesn't need the same recency
    weighting recency_weighted_touch_share applies to player shares).

    This is the opportunity-count multiplier project_td_probability
    actually needs -- team_red_zone_offense's rz_trips counts DRIVES that
    reach the red zone, not the individual plays run once there. A single
    red-zone drive averages several plays before it ends (score,
    turnover, missed FG, end of half), so using trips-per-game here
    instead of touches-per-game was found (via this model's own
    backtest) to under-predict TD probability by roughly the same ratio
    -- confirmed empirically: red-zone touches per game run ~9-11 per
    team, red-zone trips per game run ~3-4, a ~2.5-3x gap that lined up
    almost exactly with the systematic under-prediction the first
    backtest run showed across every confidence bucket."""
    prior = touch_log[touch_log["week"] < upto_week]
    if prior.empty:
        return {}
    per_game = prior.groupby(["team", "week"])["rz_touches"].sum()
    return per_game.groupby("team").mean().to_dict()


def positional_baseline_conversion(touch_log: pd.DataFrame, pos_map: dict, upto_week: int) -> dict:
    """position -> league-wide red-zone TD-per-touch rate among all
    players at that position, using games strictly before `upto_week` --
    the shrinkage target for regress_to_baseline below. Computed fresh
    per position group since RBs/WRs/TEs/QBs convert red-zone touches
    into TDs at genuinely different rates (a goal-line RB carry scores far
    more often than a red-zone WR target), so one blended league rate
    would systematically bias every position's regression target."""
    prior = touch_log[touch_log["week"] < upto_week].copy()
    if prior.empty:
        return {}
    prior["position"] = prior["player_id"].map(pos_map)
    by_pos = prior.groupby("position").agg(touches=("rz_touches", "sum"), tds=("rz_tds", "sum"))
    return {pos: (row["tds"] / row["touches"] if row["touches"] > 0 else 0.0) for pos, row in by_pos.iterrows()}


def regress_to_baseline(player_touches: float, player_tds: float, position: str, baseline_rates: dict) -> float:
    """Empirical-Bayes-style shrinkage: blends a player's own red-zone
    conversion rate (tds/touches) with the positional baseline rate,
    weighted by how many touches they actually have -- SHRINKAGE_TOUCHES
    is "how many touches of prior" the baseline is worth, so a player
    with far fewer real touches than that gets pulled hard toward the
    position average, and a player with far more barely moves off their
    own real rate. A player with zero red-zone touches yet returns the
    pure baseline rate (nothing individual to blend in)."""
    k = SHRINKAGE_TOUCHES.get(position, 8)
    baseline = baseline_rates.get(position, 0.0)
    if player_touches <= 0:
        return baseline
    return (player_touches * (player_tds / player_touches) + k * baseline) / (player_touches + k)


def _project_from_share_row(row: pd.Series, team_total: float, team: str, opponent: str, position: str,
                             team_rz_touches: dict, team_rz_defense: pd.DataFrame, baseline_rates: dict,
                             league_avg_rz_td_rate: float) -> dict | None:
    """Shared math for both project_td_probability (backtest, one season's
    own touch-share table) and project_td_probability_live (live scoring,
    resolves current-vs-fallback-season first) -- `row` is one player's
    weighted red-zone touch/TD totals, `team_total` is their team's
    weighted total those touches are a share of."""
    if team_total <= 0:
        return None
    player_share = row["w_rz_touches"] / team_total

    team_touches_per_game = team_rz_touches.get(team)
    if not team_touches_per_game:
        return None

    opp_def = team_rz_defense[team_rz_defense["team"] == opponent]
    def_factor = 1.0
    if not opp_def.empty and opp_def.iloc[0]["games"] and league_avg_rz_td_rate:
        opp_row = opp_def.iloc[0]
        opp_rz_td_rate_allowed = opp_row["rz_tds_allowed"] / opp_row["rz_trips_allowed"] if opp_row["rz_trips_allowed"] else league_avg_rz_td_rate
        def_factor = max(0.6, min(1.6, opp_rz_td_rate_allowed / league_avg_rz_td_rate))

    player_conversion = regress_to_baseline(row["w_rz_touches"], row["w_rz_tds"], position, baseline_rates)

    expected_tds = team_touches_per_game * player_share * player_conversion * def_factor
    prob = float(poisson.sf(0, expected_tds)) if expected_tds > 0 else 0.0
    return {
        "prob": prob, "expected_tds": expected_tds, "player_share": player_share,
        "player_conversion": player_conversion, "def_factor": def_factor,
        "team_rz_touches_per_game": team_touches_per_game, "n_games": int(row["n_games"]),
    }


def project_td_probability(team: str, opponent: str, player_id: str, position: str,
                            touch_share_table: pd.DataFrame, team_rz_touches: dict,
                            team_rz_defense: pd.DataFrame, baseline_rates: dict,
                            league_avg_rz_td_rate: float) -> dict | None:
    """Backtest entry point -- P(player scores >=1 TD this game), built
    from real opportunity share rather than a flat season-long rate.
    `touch_share_table` is one season's own recency_weighted_touch_share
    output (model/td_backtest.py never needs a cross-season fallback,
    since red-zone role genuinely resets with rosters/schemes each
    offseason -- see that module's docstring). None if this player has no
    red-zone touch history at all this window (nothing to project from --
    same "real limitation of leaning on prior data, not a bug" contract
    as the rest of this codebase's player-level projections)."""
    row = touch_share_table[(touch_share_table["team"] == team) & (touch_share_table["player_id"] == player_id)]
    if row.empty or row.iloc[0]["w_rz_touches"] <= 0:
        return None
    row = row.iloc[0]
    team_total = touch_share_table[touch_share_table["team"] == team]["w_rz_touches"].sum()
    return _project_from_share_row(row, team_total, team, opponent, position,
                                    team_rz_touches, team_rz_defense, baseline_rates, league_avg_rz_td_rate)


def lookup_touch_share(current: pd.DataFrame, fallback: pd.DataFrame, team: str, player_id: str,
                        min_games: int = 2) -> tuple[pd.Series, float] | None:
    """(row, team_total) from whichever of current/fallback has this
    player with >=min_games -- same current-season-if-enough-else-
    fallback contract as model/player_stats.py's _lookup_player_avg,
    applied to red-zone touch share instead of season-average stat
    lines."""
    for share in (current, fallback):
        row = share[(share["team"] == team) & (share["player_id"] == player_id)]
        if not row.empty and row.iloc[0]["n_games"] >= min_games:
            team_total = share[share["team"] == team]["w_rz_touches"].sum()
            return row.iloc[0], team_total
    return None


def project_td_probability_live(team: str, opponent: str, player_id: str, position: str,
                                 touch_share: tuple, team_rz_touches: dict, team_rz_defense: pd.DataFrame,
                                 baseline_rates: dict, league_avg_rz_td_rate: float) -> dict | None:
    """Live-scoring entry point -- `touch_share` is a (current_season,
    fallback_season) tuple of recency_weighted_touch_share tables;
    resolves per-player via lookup_touch_share before running the same
    core math project_td_probability uses. team_rz_touches/team_rz_
    defense/baseline_rates are each already merged current-with-fallback
    by the caller (model/player_stats.py's score_props()) -- team-level
    aggregates need less granularity than the strict per-player
    min-games gate red-zone SHARE needs, matching model/predict.py's
    get_pregame_stats()'s own team-level current-else-fallback merge.

    Blends in model/td_ensemble.py's trained logistic classifier here,
    not inside _project_from_share_row -- project_td_probability (the
    backtest entry point) deliberately stays pure-Poisson, a clean,
    stable baseline metric to re-check the core model against later,
    rather than measuring a moving target that includes whatever
    classifier happens to be currently saved. Falls back to the plain
    Poisson probability untouched if no classifier has been trained yet."""
    current_share, fallback_share = touch_share
    found = lookup_touch_share(current_share, fallback_share, team, player_id)
    if found is None:
        return None
    row, team_total = found
    result = _project_from_share_row(row, team_total, team, opponent, position,
                                      team_rz_touches, team_rz_defense, baseline_rates, league_avg_rz_td_rate)
    if result is None:
        return None
    result["prob"] = blend_with_classifier(result["prob"], {
        "expected_tds": result["expected_tds"], "player_share": result["player_share"],
        "player_conversion": result["player_conversion"], "def_factor": result["def_factor"],
        "team_rz_touches_per_game": result["team_rz_touches_per_game"], "n_games": result["n_games"],
        "position": position,
    })
    return result
