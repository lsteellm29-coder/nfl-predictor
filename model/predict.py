# loads model, scores this week's games
"""Loads the trained model and scores one week's games: win probability,
implied spread, and the edge against the current Vegas line.

Pre-game team stats come from the current season's games played so far. Early
in a season a team may not have played yet, in which case its stats carry
forward from the end of the previous season (there's no in-season signal yet,
so last season's final read on the team is the best available estimate).
"""

import datetime
import math
import os

import joblib
import nfl_data_py as nfl
import numpy as np
import pandas as pd
import requests

from config import CURRENT_SEASON
from data.fetch_injuries import fetch_current_injury_impact
from data.fetch_week import fetch_week
from data.fetch_weather import fetch_forecast
from data.leakage import assert_no_leakage
from data.player_trends import qb_game_log, qb_streak, rb_game_log, rb_streak
from data.positional_matchups import build_position_tables, game_mismatches, position_map
from data.rosters import fetch_rosters
from data.situational import (
    away_travel_penalty, blowout_loss_flags, is_short_week, lookahead_flags,
)
from data.team_change_tracker import (
    head_coach_changes, team_change_confidence_flag, team_unit_turnover,
)
from data.team_codes import normalize_team_codes
from data.team_history import coach_h2h, team_last_n_meetings
from data.team_stats import build_rolling_team_stats, build_team_game_stats
from model.calibration import apply_calibrator
from model.elo import compute_elo_ratings
from model.td_model import (
    player_red_zone_touches, positional_baseline_conversion, project_td_probability_live,
    recency_weighted_touch_share, season_to_date, team_red_zone_defense, team_red_zone_touches_per_game,
)
from model.train import FEATURE_COLS, STAT_COLS, predict_proba
from report.narrative import select_lead_narrative

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEAM_STATS_PATH = os.path.join(ROOT_DIR, "data", "cache", "team_stats.parquet")
PBP_PATH = os.path.join(ROOT_DIR, "data", "cache", "pbp.parquet")
SCHEDULES_PATH = os.path.join(ROOT_DIR, "data", "cache", "schedules.parquet")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")

SPLIT_COLS = ["home_point_diff_avg", "away_point_diff_avg"]
STATS_COLS = STAT_COLS + SPLIT_COLS


def _current_season_stats(season: int, week: int) -> pd.DataFrame:
    """Rolling stats built from this season's games played before `week`.
    Empty if the season hasn't started yet or no one has played."""
    schedule = nfl.import_schedules([season])
    played = schedule[
        (schedule["game_type"] == "REG")
        & schedule["home_score"].notna()
        & (schedule["week"] < week)
    ]
    if played.empty:
        return pd.DataFrame(columns=["team", *STATS_COLS])

    pbp = nfl.import_pbp_data([season], downcast=True)
    team_game_stats = build_team_game_stats(schedule, pbp)
    rolling = build_rolling_team_stats(team_game_stats)
    rolling = rolling[rolling["week"] < week]
    # Phase 2 leakage tripwire: this filter IS the actual enforcement
    # boundary for current-season rolling stats -- a separate, explicit
    # assertion right after it (rather than trusting the filter
    # expression alone) means a future refactor that accidentally
    # loosens it to `<=` fails loudly here instead of quietly training
    # on a team's own game.
    assert_no_leakage(rolling, week, context="_current_season_stats")
    return rolling.sort_values("week").groupby("team").tail(1)


def _fallback_stats(season: int) -> pd.DataFrame:
    """Each team's final rolling-stat row from the prior cached season, for
    teams with no games played yet this season."""
    historical = pd.read_parquet(TEAM_STATS_PATH)
    prior = historical[historical["season"] == season - 1]
    return prior.sort_values("week").groupby("team").tail(1)


# Combined Build Plan Part 2 (Preseason / Season-Not-Started Detection):
# _fallback_stats() above silently treats a team's final prior-season
# rolling stats as this season's estimate whenever there's no current-
# season sample yet -- reasonable as the best available number, but it
# never told a viewer that's what happened, or how much the roster
# behind that number has actually turned over since. The functions below
# build that risk signal for report/cards.py's display layer; neither
# one changes what get_pregame_stats() feeds the model itself -- this is
# strictly a "how much should a viewer trust this" caveat, not a second,
# silent adjustment to the prediction (spec's own explicit requirement).
FALLBACK_GAMES_THRESHOLD = 3  # spec: "under ~2-3 games played" counts as a thin current-season sample
MEANINGFUL_SNAP_PCT = 0.50  # "starters/high-snap, not depth-chart bottom"
MEANINGFUL_MIN_GAMES = 4
# qa/validate_rosters.py's year_over_year_changes() already documents
# normal season-over-season NFL roster churn as commonly 15-25% -- this
# has to sit clearly above that range, or the caveat would fire for
# nearly every team every offseason instead of flagging teams that have
# genuinely changed more than normal.
TURNOVER_CAVEAT_THRESHOLD = 0.35


def _normalize_name(name: str) -> str:
    """Same normalization data/fetch_firecrawl_sources.py's own copy
    applies -- nfl_data_py's snap-count table (PFR-sourced) and its
    seasonal-roster table (nflverse-sourced) don't always agree on
    suffix formatting for the same real person. Kept as a local copy
    rather than a shared import, same precedent that module's own copy
    already set for this codebase's independent-source name matching."""
    import re
    name = name.replace(".", "")
    name = re.sub(r"\s+(jr|sr|ii|iii|iv)$", "", name, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", name).strip().lower()


def _meaningful_usage_players(season: int) -> dict[str, set[str]]:
    """team -> normalized names of that team's real starters/high-snap
    players in `season` -- averaged >= MEANINGFUL_SNAP_PCT offensive OR
    defensive snap share across >= MEANINGFUL_MIN_GAMES games, nfl_data_py's
    import_snap_counts() (Pro-Football-Reference sourced). Snap share is
    a direct usage measurement, not a proxy reconstructed from touches/
    targets -- the actual "starter vs. depth chart" signal the spec asks
    for, not an approximation of it. Team codes normalized (data/
    team_codes.py) since this gets compared against fetch_rosters()'s
    already-canonicalized codes in team_roster_turnover() below -- see
    that module's docstring for the real bug an unnormalized comparison
    like this one caused before it existed."""
    snaps = normalize_team_codes(nfl.import_snap_counts([season]))
    snaps = snaps[snaps["game_type"] == "REG"]
    if snaps.empty:
        return {}
    snap_pct = snaps[["offense_pct", "defense_pct"]].max(axis=1)
    snaps = snaps.assign(snap_pct=snap_pct)
    agg = snaps.groupby(["team", "player"])["snap_pct"].agg(["mean", "size"])
    starters = agg[(agg["mean"] >= MEANINGFUL_SNAP_PCT) & (agg["size"] >= MEANINGFUL_MIN_GAMES)]

    result: dict[str, set[str]] = {}
    for (team, player), _ in starters.iterrows():
        result.setdefault(team, set()).add(_normalize_name(player))
    return result


def team_roster_turnover(season: int) -> dict[str, dict]:
    """Combined Build Plan Part 2 step 1: for each team, the share of
    LAST season's real starters/high-snap players (_meaningful_usage_players)
    who aren't on this season's CURRENT roster at all (data/rosters.py's
    fetch_rosters(), the same current-season-wins guarantee the roster-
    integrity fix established elsewhere in this build). A team with no
    real starters recorded last season (a defunct edge case, not a real
    NFL team) is simply absent from the result rather than divide-by-zero."""
    prior_starters = _meaningful_usage_players(season - 1)
    if not prior_starters:
        return {}

    current_rosters = fetch_rosters([season])
    current_by_team: dict[str, set] = {}
    for team, names in current_rosters.groupby("team")["player_name"]:
        current_by_team[team] = {_normalize_name(n) for n in names}

    result = {}
    for team, starters in prior_starters.items():
        if not starters:
            continue
        current_names = current_by_team.get(team, set())
        departed = sum(1 for p in starters if p not in current_names)
        result[team] = {
            "turnover_pct": departed / len(starters),
            "departed": departed, "total_starters": len(starters),
        }
    return result


def team_fallback_status(season: int, week: int) -> dict[str, dict]:
    """Combined Build Plan Part 2 step 2: how many real current-season
    games each team has played before `week`, and whether that count is
    thin enough (FALLBACK_GAMES_THRESHOLD) to matter for the display
    caveat. Deliberately separate from get_pregame_stats()'s own
    current-vs-fallback stat SELECTION, which already switches to
    current-season data the moment a team has even one real game (the
    right call for the model itself) -- "has any current-season data at
    all" and "has enough of it to outweigh roster-turnover risk" are two
    different questions, and this function only answers the second one.
    A team not yet in the schedule's played-games slice at all (true
    preseason, zero games) gets games_played=0 explicitly, not a missing
    key."""
    schedule = nfl.import_schedules([season])
    all_teams = set(schedule["home_team"]) | set(schedule["away_team"])
    played = schedule[(schedule["game_type"] == "REG") & schedule["home_score"].notna() & (schedule["week"] < week)]
    games_played = _team_games_played(played)
    return {
        team: {"games_played": int(games_played.get(team, 0)),
               "thin_sample": games_played.get(team, 0) < FALLBACK_GAMES_THRESHOLD}
        for team in all_teams
    }


def fallback_confidence_caveat(team: str, prior_season: int, turnover: dict, fallback_status: dict) -> dict | None:
    """Combined Build Plan Part 2 step 3: the structured caveat data (or
    None) for one team -- fires only on the intersection of "thin/no
    current-season sample" AND "meaningfully high roster turnover"
    (TURNOVER_CAVEAT_THRESHOLD, well above the 15-25% normal-churn
    range), never on either alone. A thin sample with a stable roster,
    or a lot of turnover on a team already several games into real
    form, both stay uncaveated -- this is specifically about the case
    get_pregame_stats() can't tell apart from a stable team: an old
    number standing in for a roster that's actually changed a lot.

    Returns structured data, not a formatted sentence -- this module
    only ever deals in team abbreviations, never full names (that's
    report/build_report.py's _full_name(), a display-layer concern);
    report/cards.py's fallback_caveat_text() renders the actual copy
    once it has the team's full name in hand."""
    status = fallback_status.get(team)
    churn = turnover.get(team)
    if not status or not status["thin_sample"] or not churn:
        return None
    if churn["turnover_pct"] < TURNOVER_CAVEAT_THRESHOLD:
        return None
    return {
        "prior_season": prior_season, "departed": churn["departed"],
        "total_starters": churn["total_starters"], "turnover_pct": churn["turnover_pct"],
    }


def get_current_elo_ratings(season: int) -> dict:
    """Each team's {"off", "def"} Elo ratings as of right now, computed by
    replaying every game from the start of the cached historical window
    through whatever's been played of the current season so far -- Elo has
    to be run continuously to mean anything, unlike the rolling stats
    which reset each season.

    The full current-season schedule (including future, unplayed games) is
    passed in, not filtered to a target week -- unplayed games never update
    a rating (compute_elo_ratings skips games with no score), but the
    season needs at least one row present so the season-boundary
    regression-to-the-mean actually fires even before the season's first
    game has been played."""
    historical = pd.read_parquet(SCHEDULES_PATH)
    current = nfl.import_schedules([season])
    combined = pd.concat([historical, current], ignore_index=True)
    _, ratings = compute_elo_ratings(combined)
    return ratings


def get_pregame_stats(season: int, week: int) -> pd.DataFrame:
    current = _current_season_stats(season, week)
    fallback = _fallback_stats(season)
    combined = pd.concat([
        current,
        fallback[~fallback["team"].isin(current["team"])],
    ])
    return combined.set_index("team")


def get_game_wind_speed(game: pd.Series) -> float:
    """Live forecast wind (mph) for this game's stadium, or 0 for dome/
    closed-roof games and for games too far out for a forecast to exist yet
    (nothing knowable this far ahead, same as assuming calm)."""
    if game.get("roof") not in ("outdoors", "open"):
        return 0.0
    if pd.isna(game.get("gameday")) or pd.isna(game.get("gametime")):
        return 0.0
    try:
        kickoff = datetime.datetime.strptime(f"{game['gameday']} {game['gametime']}", "%Y-%m-%d %H:%M")
    except ValueError:
        return 0.0

    try:
        forecast = fetch_forecast(game["home_team"], kickoff)
    except requests.RequestException:
        return 0.0
    return forecast["wind"] if forecast else 0.0


def get_situational_flags(season: int, week: int) -> tuple[dict, dict]:
    """(blowout_loss, lookahead) team -> flag dicts for the current season,
    same logic as training (data.situational), applied to whatever's been
    scheduled/played so far this season. A team with no prior game this
    season yet (e.g. week 1) has no signal either way -- same as how the
    historical version resets to 0 each new season, so this stays
    consistent with what the model was trained on."""
    schedule = nfl.import_schedules([season])
    blowouts = blowout_loss_flags(schedule)
    lookaheads = lookahead_flags(schedule)

    blowout_now = blowouts[blowouts["week"] == week]
    lookahead_now = lookaheads[lookaheads["week"] == week]
    return (
        dict(zip(blowout_now["team"], blowout_now["blowout_loss_last_game"])),
        dict(zip(lookahead_now["team"], lookahead_now["lookahead_spot"])),
    )


def _build_features(
    game: pd.Series, stats: pd.DataFrame, injury_impact: dict, elo_ratings: dict,
    blowout_flags: dict, lookahead_flags_: dict,
) -> dict | None:
    home, away = game["home_team"], game["away_team"]
    if home not in stats.index or away not in stats.index:
        return None
    # market_spread (Phase 6) is now a direct model input, not just a
    # post-hoc edge comparison -- a game with no line posted yet simply
    # can't be scored by this model, same "not enough data" treatment as
    # a team with no rolling stats yet.
    if pd.isna(game.get("spread_line")):
        return None
    h, a = stats.loc[home], stats.loc[away]

    feat = {f"{col}_diff": h[col] - a[col] for col in STAT_COLS}
    # Neutral-site/international games get no home-field credit -- same
    # treatment as model/train.py's build_feature_frame.
    if game.get("location") == "Neutral":
        feat["home_field_context_diff"] = 0.0
    else:
        # Week 1 Audit & Tuning Plan Phase 2: same fillna(0.0) as
        # model/train.py's build_feature_frame() -- a team with no home
        # (or no away) game yet this season has no real figure for one
        # side of this diff, which would otherwise reach the model as a
        # raw NaN and either error out or silently corrupt a live
        # prediction, not just drop a training row the way it does at
        # training time.
        home_context = h["home_point_diff_avg"] - a["away_point_diff_avg"]
        feat["home_field_context_diff"] = 0.0 if pd.isna(home_context) else home_context
    feat["rest_diff"] = game["home_rest"] - game["away_rest"]
    # A team with no key absent from the live injury feed just had nothing
    # worth listing -- 0 impact, not missing data.
    feat["injury_impact_diff"] = injury_impact.get(home, 0.0) - injury_impact.get(away, 0.0)
    home_elo = elo_ratings.get(home, {"off": 1500.0, "def": 1500.0})
    away_elo = elo_ratings.get(away, {"off": 1500.0, "def": 1500.0})
    feat["off_elo_diff"] = home_elo["off"] - away_elo["def"]
    feat["def_elo_diff"] = home_elo["def"] - away_elo["off"]
    feat["market_spread"] = game["spread_line"]
    feat["wind_speed"] = get_game_wind_speed(game)

    feat["short_week_diff"] = int(is_short_week(game["home_rest"])) - int(is_short_week(game["away_rest"]))
    feat["away_travel_penalty"] = -1 if away_travel_penalty(
        home, away, game.get("gametime"), neutral_site=game.get("location") == "Neutral") else 0
    feat["div_game"] = game.get("div_game", 0)
    feat["blowout_loss_diff"] = blowout_flags.get(home, 0) - blowout_flags.get(away, 0)
    feat["lookahead_diff"] = lookahead_flags_.get(home, 0) - lookahead_flags_.get(away, 0)
    return feat


def _team_games_played(schedule: pd.DataFrame) -> dict:
    """team -> number of completed REG games in the given schedule slice."""
    reg = schedule[(schedule["game_type"] == "REG") & schedule["home_score"].notna()]
    home = reg.groupby("home_team").size()
    away = reg.groupby("away_team").size()
    return home.add(away, fill_value=0).to_dict()


def build_td_scorer_inputs(season: int, week: int):
    """Everything get_td_scorer_prediction needs, built once per
    score_week() call and reused across every game -- the same
    red-zone-share model/td_model.py inputs model/player_stats.py's
    score_props() builds for the props layer, so the game-breakdown
    "likely TD scorer" callout and the props cards agree with each other
    instead of running on two different, disagreeing models (this used
    to be a much cruder "whoever has the most TDs so far this season"
    count with no real opportunity signal behind it)."""
    fallback_pbp = pd.read_parquet(PBP_PATH)
    fallback_pbp = fallback_pbp[fallback_pbp["season"] == season - 1]
    current_pbp = _current_season_pbp_for_td(season, week, fallback_pbp)
    pos_map = position_map([season, season - 1])

    touch_log_current = player_red_zone_touches(current_pbp)
    touch_log_fallback = player_red_zone_touches(fallback_pbp)
    touch_share = (
        recency_weighted_touch_share(touch_log_current, week),
        recency_weighted_touch_share(touch_log_fallback, 30),
    )
    team_touches = {
        **team_red_zone_touches_per_game(touch_log_fallback, 30),
        **team_red_zone_touches_per_game(touch_log_current, week),
    }
    rz_def_current = season_to_date(team_red_zone_defense(current_pbp), ["team"], ["rz_trips_allowed", "rz_tds_allowed"], week)
    rz_def_fallback = season_to_date(team_red_zone_defense(fallback_pbp), ["team"], ["rz_trips_allowed", "rz_tds_allowed"], 30)
    team_def = pd.concat([rz_def_fallback[~rz_def_fallback["team"].isin(rz_def_current["team"])], rz_def_current])
    baseline_rates = {
        **positional_baseline_conversion(touch_log_fallback, pos_map, 30),
        **positional_baseline_conversion(touch_log_current, pos_map, week),
    }
    league_avg_rz_td_rate = (
        (team_def["rz_tds_allowed"] / team_def["rz_trips_allowed"]).mean() if not team_def.empty else None)

    # Candidate pool per team: anyone with real red-zone touch history
    # (current or fallback season) -- simpler than reusing
    # model/player_stats.py's required_lineup ranking (a display-coverage
    # concern, not relevant here) and directly matches "who's actually
    # gotten real scoring chances."
    rosters = fetch_rosters([season, season - 1])
    id_to_name = dict(zip(rosters["player_id"], rosters["player_name"]))
    candidates_by_team: dict[str, set] = {}
    for share_table in touch_share:
        for team, player_id in zip(share_table["team"], share_table["player_id"]):
            candidates_by_team.setdefault(team, set()).add(player_id)

    return {
        "touch_share": touch_share, "team_touches": team_touches, "team_def": team_def,
        "baseline_rates": baseline_rates, "league_avg_rz_td_rate": league_avg_rz_td_rate,
        "pos_map": pos_map, "id_to_name": id_to_name, "candidates_by_team": candidates_by_team,
    }


def _current_season_pbp_for_td(season: int, week: int, fallback_pbp: pd.DataFrame) -> pd.DataFrame:
    schedule = nfl.import_schedules([season])
    played = schedule[(schedule["game_type"] == "REG") & schedule["home_score"].notna() & (schedule["week"] < week)]
    if played.empty:
        return fallback_pbp.iloc[0:0]
    pbp = nfl.import_pbp_data([season], downcast=True)
    return pbp[pbp["week"] < week]


def get_td_scorer_prediction(team: str, opponent: str, td_inputs: dict,
                              implied_totals: dict, league_avg_implied_total: float) -> dict | None:
    """The team's most likely TD scorer this game -- whoever among real
    red-zone-touch candidates on `team` has the highest projected
    probability (model/td_model.py's project_td_probability_live, same
    engine the props cards use), with one more adjustment the props layer
    doesn't apply: this game's Vegas-implied team total vs. the week's
    average (a shootout raises everyone's odds; a low-scoring line lowers
    them), capped 0.6x-1.6x same as the rest of this codebase's small-
    sample matchup factors."""
    candidates = td_inputs["candidates_by_team"].get(team, set())
    best_id, best_result = None, None
    for player_id in candidates:
        position = td_inputs["pos_map"].get(player_id)
        result = project_td_probability_live(
            team, opponent, player_id, position, td_inputs["touch_share"], td_inputs["team_touches"],
            td_inputs["team_def"], td_inputs["baseline_rates"], td_inputs["league_avg_rz_td_rate"])
        if result is None:
            continue
        if best_result is None or result["expected_tds"] > best_result["expected_tds"]:
            best_id, best_result = player_id, result
    if best_result is None:
        return None

    total_factor = 1.0
    if league_avg_implied_total and team in implied_totals:
        total_factor = _clip(implied_totals[team] / league_avg_implied_total, 0.6, 1.6)

    adjusted_expected_tds = best_result["expected_tds"] * total_factor
    prob = 1 - math.exp(-adjusted_expected_tds) if adjusted_expected_tds > 0 else 0.0
    return {
        "player": td_inputs["id_to_name"].get(best_id, best_id),
        "prob": prob, "base_prob": best_result["prob"],
        "def_factor": best_result["def_factor"], "total_factor": total_factor,
        "player_share": best_result["player_share"], "games": best_result["n_games"],
    }


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def get_implied_team_totals(games: pd.DataFrame) -> dict:
    """Vegas-implied points for each team in this week's games, derived from
    total_line and spread_line (home team's favored margin): home = (total +
    spread) / 2, away = (total - spread) / 2."""
    totals = {}
    for _, g in games.iterrows():
        if pd.isna(g.get("total_line")) or pd.isna(g.get("spread_line")):
            continue
        totals[g["home_team"]] = (g["total_line"] + g["spread_line"]) / 2
        totals[g["away_team"]] = (g["total_line"] - g["spread_line"]) / 2
    return totals


def _logistic_contributions(model, feat_df: pd.DataFrame) -> dict:
    """Each feature's signed contribution to the home-win logit: the
    standardized value the logistic regression actually saw, times its
    learned coefficient. This is what the model actually did, not an
    assumption about which raw stat direction is 'good' (see the
    def_epa_per_play_avg_diff sign flip caught during training -- collinear
    features can have counter-intuitive individual coefficients)."""
    scaler = model.named_steps["standardscaler"]
    logreg = model.named_steps["logisticregression"]
    scaled = scaler.transform(feat_df)[0]
    return dict(zip(feat_df.columns, scaled * logreg.coef_[0]))


def _xgboost_contributions(model, feat_df: pd.DataFrame) -> dict:
    """Exact per-feature SHAP contributions to the predicted margin, via
    XGBoost's own TreeExplainer-equivalent (pred_contribs=True) -- this is
    the real decomposition of what the trees did, not an approximation.
    Coefficients don't exist for a tree ensemble, so this replaces
    _logistic_contributions for that model type."""
    import xgboost

    booster = model.get_booster()
    contribs = booster.predict(xgboost.DMatrix(feat_df), pred_contribs=True)[0]
    return dict(zip(feat_df.columns, contribs[:-1]))  # last column is the bias term


def _feature_contributions(model_type: str, logistic_model, xgb_model, feat_df: pd.DataFrame) -> dict:
    if model_type == "xgboost":
        return _xgboost_contributions(xgb_model, feat_df)
    if model_type == "ensemble":
        # Both halves score on the same pre-sigmoid log-odds scale --
        # XGBoost's binary:logistic SHAP contributions are margin-space,
        # same as the logistic half's standardized-value * coefficient --
        # so averaging them per feature mirrors how score_week() averages
        # the two models' probabilities. Only using the logistic half here
        # would hide whatever the tree model uniquely picked up on (e.g. a
        # nonlinear injury or Elo effect), which defeats the point of
        # explaining what the ensemble actually did.
        logistic_contribs = _logistic_contributions(logistic_model, feat_df)
        xgb_contribs = _xgboost_contributions(xgb_model, feat_df)
        return {feat: (logistic_contribs[feat] + xgb_contribs[feat]) / 2 for feat in logistic_contribs}
    return _logistic_contributions(logistic_model, feat_df)


def _fallback_season_pbp(season: int) -> pd.DataFrame:
    pbp = pd.read_parquet(PBP_PATH)
    return pbp[pbp["season"] == season - 1]


def _current_season_pbp(season: int, week: int, fallback_pbp: pd.DataFrame) -> pd.DataFrame:
    """This season's pbp through `week`, or an empty frame shaped like
    `fallback_pbp` if the season hasn't produced any games yet -- same
    empty-early-season handling as _current_season_pbp_for_td."""
    schedule = nfl.import_schedules([season])
    played = schedule[
        (schedule["game_type"] == "REG") & schedule["home_score"].notna() & (schedule["week"] < week)
    ]
    if played.empty:
        return fallback_pbp.iloc[0:0]
    pbp = nfl.import_pbp_data([season], downcast=True)
    return pbp[pbp["week"] < week]


def score_week(week: int, season: int = CURRENT_SEASON) -> pd.DataFrame:
    saved = joblib.load(MODEL_PATH)
    logistic_model = saved["logistic_model"]
    xgb_model = saved["xgb_model"]
    model_type = saved.get("model_type", "logistic")
    spread_calibration = saved["spread_calibration"]
    # From model/calibration.py's audit -- only set to something other than
    # None if a calibrator demonstrably beat the raw model on a true
    # holdout, not just fit-and-eyeballed on the same data.
    calibrator = saved.get("calibrator")

    games = fetch_week(week, season)
    stats = get_pregame_stats(season, week)
    elo_ratings = get_current_elo_ratings(season)
    blowout_flags, lookahead_flags_now = get_situational_flags(season, week)

    # Combined Build Plan Part 2: computed once per call, same as stats/
    # elo_ratings above -- both are whole-league lookups, not per-game.
    try:
        roster_turnover = team_roster_turnover(season)
        fallback_status = team_fallback_status(season, week)
    except Exception as e:
        # Best-effort display caveat, never allowed to block scoring the
        # week itself over e.g. a snap-count fetch hiccup -- same fail-
        # open contract as this function's existing injury-fetch guard
        # just above.
        print(f"Warning: couldn't compute roster-turnover confidence caveat ({e}); skipping it this run.")
        roster_turnover, fallback_status = {}, {}

    try:
        injury_impact = fetch_current_injury_impact()
    except requests.RequestException as e:
        print(f"Warning: couldn't fetch live injury data ({e}); scoring without it.")
        injury_impact = {}

    td_inputs = build_td_scorer_inputs(season, week)

    implied_totals = get_implied_team_totals(games)
    league_avg_implied_total = (
        sum(implied_totals.values()) / len(implied_totals) if implied_totals else 0.0
    )

    # Section 8 narrative-lookup layer (data/positional_matchups.py,
    # data/player_trends.py, data/team_history.py) -- built once per week,
    # not per game, since it's the same underlying season data for every
    # matchup. None of this feeds FEATURE_COLS; report/narrative.py only
    # ever uses it to explain a pick already made below, never to make one.
    fallback_pbp = _fallback_season_pbp(season)
    current_pbp = _current_season_pbp(season, week, fallback_pbp)
    pos_map = {**position_map([season - 1]), **position_map([season])}
    current_pos_tables = build_position_tables(current_pbp, pos_map)
    fallback_pos_tables = build_position_tables(fallback_pbp, pos_map)
    current_qb_log, fallback_qb_log = qb_game_log(current_pbp), qb_game_log(fallback_pbp)
    current_rb_log, fallback_rb_log = rb_game_log(current_pbp, pos_map), rb_game_log(fallback_pbp, pos_map)
    historical_schedule = pd.read_parquet(SCHEDULES_PATH)
    combined_schedule = pd.concat([historical_schedule, nfl.import_schedules([season])], ignore_index=True)

    # Combined Build Plan Part 3: coaching changes + per-unit roster
    # continuity, same narrative-lookup-layer treatment (built once per
    # week, never touches FEATURE_COLS) as the block above it.
    try:
        coach_changes = head_coach_changes(season)
        unit_turnover_by_team = team_unit_turnover(season)
        team_changes = {
            team: {"coach_change": coach_changes.get(team), "unit_turnover": unit_turnover_by_team.get(team, {})}
            for team in set(coach_changes) | set(unit_turnover_by_team)
        }
    except Exception as e:
        print(f"Warning: couldn't compute team-change tracking ({e}); skipping it this run.")
        team_changes = {}

    rows = []
    for _, game in games.iterrows():
        feat = _build_features(game, stats, injury_impact, elo_ratings, blowout_flags, lookahead_flags_now)
        row = game.to_dict()
        row["wind_speed"] = feat["wind_speed"] if feat else get_game_wind_speed(game)
        row["home_fallback_caveat"] = fallback_confidence_caveat(
            game["home_team"], season - 1, roster_turnover, fallback_status)
        row["away_fallback_caveat"] = fallback_confidence_caveat(
            game["away_team"], season - 1, roster_turnover, fallback_status)

        # Combined Build Plan Part 3 step 4: the SAME confidence-caveat
        # system Part 2 established, triggered by a different signal --
        # a new head coach plus meaningfully high position-unit turnover,
        # active through a longer games-played window than Part 2's own
        # (COACH_CHANGE_GAMES_WINDOW vs. FALLBACK_GAMES_THRESHOLD -- a
        # new scheme takes longer to prove itself than a thin stat sample
        # alone does).
        home_change = team_changes.get(game["home_team"], {})
        away_change = team_changes.get(game["away_team"], {})
        home_games_played = fallback_status.get(game["home_team"], {}).get("games_played", 0)
        away_games_played = fallback_status.get(game["away_team"], {}).get("games_played", 0)
        row["home_team_change_flag"] = team_change_confidence_flag(
            home_games_played, home_change.get("coach_change"), home_change.get("unit_turnover", {}))
        row["away_team_change_flag"] = team_change_confidence_flag(
            away_games_played, away_change.get("coach_change"), away_change.get("unit_turnover", {}))

        # Combined Build Plan Part 2 step 3: nudge the DISPLAYED confidence
        # tier, never the model's own probability -- report/cards.py's
        # confidence_tier() takes this as an extra "uncertain" flag that
        # forces the toss-up color regardless of how confident the raw
        # number looks, same "don't silently adjust, don't silently trust
        # either" split the spec asks for. Part 3's team-change flag folds
        # into the exact same displayed-tier nudge, not a second separate
        # mechanism.
        row["confidence_uncertain"] = bool(
            row["home_fallback_caveat"] or row["away_fallback_caveat"]
            or row["home_team_change_flag"] or row["away_team_change_flag"])
        row["home_td_scorer"] = get_td_scorer_prediction(
            game["home_team"], game["away_team"], td_inputs, implied_totals, league_avg_implied_total)
        row["away_td_scorer"] = get_td_scorer_prediction(
            game["away_team"], game["home_team"], td_inputs, implied_totals, league_avg_implied_total)
        default_elo = {"off": 1500.0, "def": 1500.0}
        if game["home_team"] in stats.index:
            row["home_stats"] = stats.loc[game["home_team"]].to_dict()
            row["home_stats"]["injury_impact"] = injury_impact.get(game["home_team"], 0.0)
            home_elo = elo_ratings.get(game["home_team"], default_elo)
            row["home_stats"]["off_elo"] = home_elo["off"]
            row["home_stats"]["def_elo"] = home_elo["def"]
        if game["away_team"] in stats.index:
            row["away_stats"] = stats.loc[game["away_team"]].to_dict()
            row["away_stats"]["injury_impact"] = injury_impact.get(game["away_team"], 0.0)
            away_elo = elo_ratings.get(game["away_team"], default_elo)
            row["away_stats"]["off_elo"] = away_elo["off"]
            row["away_stats"]["def_elo"] = away_elo["def"]
        row["home_blowout_loss"] = blowout_flags.get(game["home_team"], 0)
        row["away_blowout_loss"] = blowout_flags.get(game["away_team"], 0)
        row["home_lookahead"] = lookahead_flags_now.get(game["home_team"], 0)
        row["away_lookahead"] = lookahead_flags_now.get(game["away_team"], 0)
        if feat is None:
            row["home_win_prob"] = None
            row["implied_spread"] = None
            row["edge"] = None
            row["top_factors"] = None
            row["lead_narrative"] = None
            rows.append(row)
            continue

        feat_df = pd.DataFrame([feat])[FEATURE_COLS]
        raw_win_prob = predict_proba(model_type, logistic_model, xgb_model, feat_df)[0, 1]
        home_win_prob = float(apply_calibrator(calibrator, np.array([raw_win_prob]))[0])
        implied_spread = spread_calibration.predict([[home_win_prob]])[0]

        row["home_win_prob"] = home_win_prob
        row["implied_spread"] = implied_spread
        # positive edge = model favors the home team more than the market does
        row["edge"] = implied_spread - game["spread_line"] if pd.notna(game["spread_line"]) else None

        contributions = _feature_contributions(model_type, logistic_model, xgb_model, feat_df)
        row["top_factors"] = sorted(contributions.items(), key=lambda kv: -abs(kv[1]))[:3]

        home, away = game["home_team"], game["away_team"]
        winner, loser = (home, away) if home_win_prob >= 0.5 else (away, home)
        mismatches = game_mismatches(home, away, current_pos_tables, fallback_pos_tables)
        qb_streaks = [
            qb_streak(home, game.get("home_qb_id"), current_qb_log, fallback_qb_log),
            qb_streak(away, game.get("away_qb_id"), current_qb_log, fallback_qb_log),
        ]
        rb_streaks = [
            rb_streak(home, current_rb_log, fallback_rb_log),
            rb_streak(away, current_rb_log, fallback_rb_log),
        ]
        team_h2h = team_last_n_meetings(home, away, combined_schedule)
        coach_h2h_record = coach_h2h(game.get("home_coach"), game.get("away_coach"), combined_schedule)
        row["lead_narrative"] = select_lead_narrative(
            winner, loser, mismatches, qb_streaks, rb_streaks, team_h2h, coach_h2h_record, team_changes)
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--season", type=int, default=CURRENT_SEASON)
    args = parser.parse_args()

    preds = score_week(args.week, args.season)
    cols = ["away_team", "home_team", "home_win_prob", "spread_line",
            "implied_spread", "edge"]
    with pd.option_context("display.float_format", "{:.2f}".format):
        print(preds[cols].to_string(index=False))


if __name__ == "__main__":
    main()
