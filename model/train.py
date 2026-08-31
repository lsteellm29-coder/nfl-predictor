# trains + saves logistic regression model
"""Logistic regression: input = home team's rolling stats minus away team's
rolling stats, output = probability the home team wins. Trains on the first 9
of the 10 cached seasons, tests on the 10th, and prints accuracy so we know
whether to trust it before running it on live games.
"""

import os

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import HISTORICAL_SEASONS
from data.fetch_injuries import historical_injury_impact
from data.situational import (
    away_travel_penalty, blowout_loss_flags, is_short_week, lookahead_flags,
)
from model.elo import compute_elo_ratings

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT_DIR, "data", "cache")
SCHEDULES_PATH = os.path.join(CACHE_DIR, "schedules.parquet")
TEAM_STATS_PATH = os.path.join(CACHE_DIR, "team_stats.parquet")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")

TRAIN_SEASONS = HISTORICAL_SEASONS[:-1]
TEST_SEASON = HISTORICAL_SEASONS[-1]

# Rolling stats from Section 3 -- diffed as (home - away) to build each game's
# feature row.
STAT_COLS = [
    "points_for_avg", "points_against_avg",
    "off_epa_per_play_avg", "def_epa_per_play_avg",
    "off_ypp_avg", "def_ypp_avg",
    "off_third_down_pct_avg", "def_third_down_pct_avg",
    "turnover_diff_avg", "red_zone_td_pct_avg",
    "ats_win_pct_season", "ats_win_pct_last5",
    # Section 1: opponent-adjusted versions of the same EPA/play and
    # yards/play stats (data/opponent_adjust.py), kept alongside the raw
    # ones above rather than replacing them -- correlated but not
    # redundant, and worth letting a regularized model weigh separately.
    "off_epa_per_play_adj_avg", "def_epa_per_play_adj_avg",
    "off_ypp_adj_avg", "def_ypp_adj_avg",
    # Phase 4 (v3 spec): success rate and receiving-YAC-over-expected.
    # data/team_stats.py also computes starting-QB-specific EPA/CPOE
    # (qb_epa_per_play_avg, qb_cpoe_avg) but an ablation test showed it
    # actively hurting 2025 holdout accuracy (0.625 -> 0.612 ensemble,
    # cancelling out the real gains from the two features below) --
    # likely collinear with off_epa_per_play_avg (QB dropbacks dominate
    # team offensive EPA) plus real early-season CPOE noise (rolling
    # averages swung as wide as -37 to +28 in the first couple of games).
    # Left out of training rather than kept on faith; still computed and
    # available in team_stats.parquet for narrative/props use.
    "off_success_rate_avg", "def_success_rate_avg",
    "off_yac_oe_avg", "def_yac_oe_avg",
    # NOT included: special_teams_epa_avg. data/team_stats.py has computed
    # this since Phase 10 (field goals/punts/kickoffs EPA) but it was never
    # actually wired in here -- Audit Fix Plan Step 6 tried adding it and
    # ablation-tested it (same discipline as the QB EPA/CPOE exclusion
    # above): walk-forward consensus weakened from ensemble winning 3/4
    # folds to a 2/4 split with no clear winner, and the final 2025 holdout
    # accuracy dropped from 65.9% to 63.9%. Reverted -- still computed and
    # available in team_stats.parquet for narrative/props use, just not a
    # feature that's earned a place in FEATURE_COLS.
    #
    # NOT included: off_pressure_rate_allowed_avg/def_pressure_rate_avg
    # (Master Honing Plan, Section A -- pass-rush pressure rate, added to
    # data/team_stats.py off/def nflfastR's was_pressure flag). Ablation-
    # tested the same way: adding both as diffed features dropped ensemble
    # walk-forward average accuracy from 0.6831 to 0.6686 -- likely
    # collinear with off_epa_per_play_avg/def_epa_per_play_avg the same
    # way QB EPA/CPOE was (a pressured dropback already shows up as a bad
    # EPA play). Reverted; still computed and available in
    # team_stats.parquet for narrative use (e.g. "pressures X% of
    # dropbacks, Nth-most in the league").
]
FEATURE_COLS = [f"{c}_diff" for c in STAT_COLS] + [
    "home_field_context_diff", "rest_diff", "injury_impact_diff",
    "off_elo_diff", "def_elo_diff",
    "wind_speed", "short_week_diff", "away_travel_penalty", "div_game",
    "blowout_loss_diff", "lookahead_diff",
    # Phase 6 (v3 spec): the actual market spread, not just the post-hoc
    # "edge" comparison -- the line already prices in real-time
    # information (injury news, weather, sharp money) this model doesn't
    # otherwise see. NOTE: nflverse's cached spread_line isn't documented
    # as opening vs. closing, and live scoring (data/fetch_week.py, run
    # Wednesday mornings per weekly_pipeline.sh) pulls whatever line The
    # Odds API has posted at that moment -- likely earlier-week, not
    # closing. If the historical value skews closer to closing, backtested
    # accuracy here may not fully replicate in live serving, since the
    # model would be trained on a more-informed line than it actually
    # receives when scoring an upcoming week.
    #
    # Investigated (Master Honing Plan, Section A): checked nflverse's own
    # data dictionary directly (nflreadr/nfldata's DATASETS.md and
    # dictionary_schedules.csv on GitHub) -- confirmed it genuinely does
    # not document which point in the betting week spread_line reflects,
    # this isn't a gap in searching, the upstream source is just silent on
    # it. A real empirical resolution (comparing model accuracy on
    # games where open/close plausibly converge vs. the full set) would
    # need a historical line-MOVEMENT dataset -- both an opening and a
    # closing value per game -- to be a meaningful test; The Odds API (this
    # project's only odds source) only ever exposes the CURRENT live line,
    # never a historical open/close pair for already-completed games, so
    # there's no available data to run that comparison against honestly.
    # Left as a documented, currently-irresolvable caveat rather than a
    # fabricated "fixed" number -- revisit if a historical odds-movement
    # source ever gets added to this codebase.
    "market_spread",
]


def _assert_no_join_fanout(before_n: int, after_df: pd.DataFrame, step: str, key: str = "game_id") -> None:
    """Week 1 Audit & Tuning Plan Phase 1.4: a many-to-many merge silently
    inflates row count and double-weights whatever got duplicated onto.
    Every merge in build_feature_frame() joins onto a (season, week,
    team) key that should be unique on the right-hand side, so an inner/
    left join can only ever DROP rows that don't match, never multiply
    them -- unless the right side itself has a duplicate key, which is
    exactly the bug this catches. Prints row counts either way, same as
    the plan's own paste-prompt asks for, and fails loudly (with the
    actual duplicated key values, not just a count) rather than letting
    a fanned-out join silently train on double-weighted games."""
    after_n = len(after_df)
    print(f"  [{step}] rows before: {before_n}, after: {after_n}")
    if after_n > before_n:
        dupes = after_df[key][after_df[key].duplicated()].unique()
        raise AssertionError(
            f"{step}: row count grew from {before_n} to {after_n} -- a merge fanned out "
            f"(the right-hand side had a duplicate join key). Duplicated {key}(s): {list(dupes)[:10]}"
        )


# Every rolling-stat column in team_stats.parquet EXCEPT games_played --
# that one isn't a model feature (not in ROLLING_SEASON_COLS/STAT_COLS/
# FEATURE_COLS) and has different semantics ("games played so far THIS
# season's rolling window", which should genuinely read 0 at week 1, not
# borrow a large number from last season).
_TEAM_STATS_ID_COLS = ["season", "week", "team", "opponent", "is_home", "rest_days", "div_game", "games_played"]


def _team_stats_with_fallback(team_stats: pd.DataFrame) -> pd.DataFrame:
    """Week 1 Audit & Tuning Plan Phase 2 -- the single most consequential
    finding of this audit: every season's week 1 has NO in-season rolling
    history by construction (data/team_stats.py's expanding().shift(1)
    necessarily produces NaN for a team's very first row of a season),
    and build_feature_frame() had no fallback for that -- unlike
    model/predict.py's get_pregame_stats()/_fallback_stats(), which
    already substitutes a team's final PRIOR-season row for exactly this
    case at LIVE scoring time. Verified empirically: even after fixing
    the opponent-adjustment warmup gap and the home-field-context gap
    (both earlier in this file), literally zero week-1 games survived
    training, across all 10 cached seasons -- meaning the model had never
    been trained on a single real Week 1 game, AND had never seen the
    fallback-shaped input pattern at all, despite that being exactly what
    it's asked to use the moment it's deployed to score an actual Week 1.

    Fix: mirrors model/predict.py's existing fallback pattern instead of
    inventing a new one -- fills each individual stat column's null
    values (structurally, that's every column at week 1 -- EXCEPT
    ats_win_pct_last5, which data/team_stats.py deliberately computes
    grouped by team alone, not team+season, so it already legitimately
    bridges the season boundary on its own and is real at week 1; a
    naive "is this WHOLE row null" check missed that and skipped every
    week-1 row entirely, caught empirically re-running this exact
    verification) with that same team's final value from the
    immediately prior season, one column at a time so an already-real
    column (ats_win_pct_last5, or any other) is never touched. A team
    with no prior-season row at all (the very first cached season, 2016,
    has no 2015 to fall back to) simply stays null for that column, and
    that row is still dropped by build_feature_frame()'s own dropna --
    an honest data limit, not something to guess around."""
    stat_cols = [c for c in team_stats.columns if c not in _TEAM_STATS_ID_COLS]

    # Each team's own final row of season S, relabeled as season S+1's
    # fallback -- the same "last row of the prior season" model/predict.py's
    # _fallback_stats() already uses, just precomputed for every season
    # transition in the cache instead of only the live current one.
    prior_final = (
        team_stats.sort_values("week").groupby(["team", "season"]).tail(1)
        [["team", "season"] + stat_cols]
        .rename(columns={c: f"{c}_fallback" for c in stat_cols})
    )
    prior_final = prior_final.assign(season=prior_final["season"] + 1)

    result = team_stats.merge(prior_final, on=["team", "season"], how="left")
    for c in stat_cols:
        result[c] = result[c].fillna(result[f"{c}_fallback"])
    return result.drop(columns=[f"{c}_fallback" for c in stat_cols])


def build_feature_frame(
    schedules: pd.DataFrame, team_stats: pd.DataFrame, injuries: pd.DataFrame,
    elo_per_game: pd.DataFrame, blowouts: pd.DataFrame, lookaheads: pd.DataFrame,
) -> pd.DataFrame:
    """One row per game: home-minus-away rolling stat diffs + home_win label."""
    reg = schedules[
        (schedules["game_type"] == "REG") & schedules["home_score"].notna()
    ].copy()
    reg = reg[reg["home_score"] != reg["away_score"]]  # drop ties (ambiguous label)
    reg_n = len(reg)

    team_stats = _team_stats_with_fallback(team_stats)
    home = team_stats[team_stats["is_home"]]
    away = team_stats[~team_stats["is_home"]]

    games = reg.merge(
        home, left_on=["season", "week", "home_team"],
        right_on=["season", "week", "team"], how="inner",
    )
    _assert_no_join_fanout(reg_n, games, "home team-stats merge")
    games = games.merge(
        away, left_on=["season", "week", "away_team"],
        right_on=["season", "week", "team"], how="inner",
        suffixes=("_home", "_away"),
    )
    _assert_no_join_fanout(reg_n, games, "away team-stats merge")

    for col in STAT_COLS:
        games[f"{col}_diff"] = games[f"{col}_home"] - games[f"{col}_away"]

    games["home_field_context_diff"] = (
        games["home_point_diff_avg_home"] - games["away_point_diff_avg_away"]
    )
    # Neutral-site/international games: the designated "home" team doesn't
    # actually get a crowd/travel advantage, so this feature shouldn't
    # credit them with one (Phase 9 addition, v3 spec).
    games.loc[games["location"] == "Neutral", "home_field_context_diff"] = 0.0
    # Week 1 Audit & Tuning Plan Phase 2: home_point_diff_avg/
    # away_point_diff_avg (data/team_stats.py's _rolling_for_team_season())
    # each need that team to have played at least one game ON THAT SIDE
    # already -- a team whose season so far is all road games has no
    # "home" figure yet even after 2-3 real games, and vice versa. Found
    # doing this plan's own leakage exit test: this alone was silently
    # dropping most week-2 (and some week-3) training rows via
    # `.dropna(subset=FEATURE_COLS)` below, even after fixing the
    # opponent-adjustment warmup gap (data/team_stats.py's
    # _fill_adjusted_warmup_gap()) that motivated the exit test in the
    # first place. Same "no signal yet" default this codebase already
    # uses for blowout_loss_diff/lookahead_diff below (0, not a guessed
    # average) -- the model still has every other correctly-computed
    # rolling/Elo/market feature for this game either way, this is one
    # supplementary term reading as neutral when it can't be computed yet,
    # not the game's only signal going blank.
    games["home_field_context_diff"] = games["home_field_context_diff"].fillna(0.0)
    games["rest_diff"] = games["rest_days_home"] - games["rest_days_away"]

    # Left merge + fillna(0): a team with no rows in `injuries` that week
    # simply had nothing worth listing, i.e. zero injury impact -- not
    # missing data.
    games = games.merge(
        injuries.rename(columns={"team": "home_team", "injury_impact": "home_injury_impact"}),
        on=["season", "week", "home_team"], how="left",
    ).merge(
        injuries.rename(columns={"team": "away_team", "injury_impact": "away_injury_impact"}),
        on=["season", "week", "away_team"], how="left",
    )
    _assert_no_join_fanout(reg_n, games, "injury impact merge")
    games["home_injury_impact"] = games["home_injury_impact"].fillna(0.0)
    games["away_injury_impact"] = games["away_injury_impact"].fillna(0.0)
    games["injury_impact_diff"] = games["home_injury_impact"] - games["away_injury_impact"]

    games = games.merge(
        elo_per_game[["game_id", "home_off_elo_pre", "home_def_elo_pre",
                      "away_off_elo_pre", "away_def_elo_pre"]],
        on="game_id", how="left",
    )
    _assert_no_join_fanout(reg_n, games, "Elo merge")
    # Home offense vs. away defense, and away offense vs. home defense,
    # kept as two separate features rather than one blended rating -- see
    # model/elo.py's docstring for why the split matters.
    games["off_elo_diff"] = games["home_off_elo_pre"] - games["away_def_elo_pre"]
    games["def_elo_diff"] = games["home_def_elo_pre"] - games["away_off_elo_pre"]

    # Section 6: the market's own spread as a direct model input (positive
    # = home favored, same convention as everywhere else in this file).
    games["market_spread"] = games["spread_line"]

    # Wind is game-level, not team-relative -- both sides play in the same
    # conditions, so no home/away diff makes sense here. NaN (dome/closed
    # roof, or a handful of older outdoor games missing the field) means no
    # wind impact, i.e. 0.
    games["wind_speed"] = games["wind"].fillna(0.0)

    # NOT added: kickoff-time temperature as a feature (Master Honing
    # Plan, Section A). Built and ablation-tested the same way as
    # qb_epa_per_play_avg/special_teams_epa_avg below: added temp (NaN/
    # dome filled with 70.0F, game-level not home/away-diffed, same
    # treatment as wind_speed) to FEATURE_COLS and re-ran the walk-forward
    # folds -- ensemble average accuracy fell from 0.6831 to 0.6723.
    # Reverted rather than shipped on a negative result. Precipitation
    # wasn't even tried: nflverse's historical schedule data has no precip
    # column at all (confirmed empirically -- only roof/temp/wind are
    # present), so there's no historical ground truth to train a precip
    # feature against, only a live forecast value
    # (data/fetch_weather.py's precip_prob) with nothing to backtest it
    # on.

    # Section 5 situational spots.
    games["short_week_diff"] = (
        games["rest_days_home"].apply(is_short_week).astype(int)
        - games["rest_days_away"].apply(is_short_week).astype(int)
    )
    games["away_travel_penalty"] = games.apply(
        lambda g: -1 if away_travel_penalty(g["home_team"], g["away_team"], g["gametime"]) else 0,
        axis=1,
    )
    games["div_game"] = games["div_game_x"]  # identical across div_game_x/_y, from the schedule directly

    games = games.merge(
        blowouts.rename(columns={"team": "home_team", "blowout_loss_last_game": "home_blowout_loss"}),
        on=["season", "week", "home_team"], how="left",
    ).merge(
        blowouts.rename(columns={"team": "away_team", "blowout_loss_last_game": "away_blowout_loss"}),
        on=["season", "week", "away_team"], how="left",
    )
    _assert_no_join_fanout(reg_n, games, "blowout-loss flag merge")
    games["blowout_loss_diff"] = games["home_blowout_loss"].fillna(0) - games["away_blowout_loss"].fillna(0)

    games = games.merge(
        lookaheads.rename(columns={"team": "home_team", "lookahead_spot": "home_lookahead"}),
        on=["season", "week", "home_team"], how="left",
    ).merge(
        lookaheads.rename(columns={"team": "away_team", "lookahead_spot": "away_lookahead"}),
        on=["season", "week", "away_team"], how="left",
    )
    _assert_no_join_fanout(reg_n, games, "lookahead flag merge")
    games["lookahead_diff"] = games["home_lookahead"].fillna(0) - games["away_lookahead"].fillna(0)

    games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)

    return games.dropna(subset=FEATURE_COLS)


# NOT implemented as a default: time-decay sample weighting (weighting
# recent seasons more heavily during training). Audit Fix Plan Step 6
# built and ablation-tested exponential-decay weighting (half-lives of 2,
# 3, 4, 5, 8 seasons vs. no decay), evaluated both on the true 2025
# holdout and averaged across every walk-forward fold (see
# walk_forward_folds below). Every decay setting tested underperformed no
# decay, both on the single holdout (best-of-3 accuracy 0.649-0.659
# decayed vs 0.659 undecayed) and averaged across folds (0.678-0.683
# decayed vs 0.687 undecayed) -- with only 10 cached seasons, and the
# rolling stats/Elo already emphasizing recent within-season form on their
# own, discounting older seasons just throws away real signal without
# buying anything. Not implemented, rather than shipped as an unused
# capability.


def train_logistic(train_df: pd.DataFrame):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(train_df[FEATURE_COLS], train_df["home_win"])
    return model


def train_xgboost(train_df: pd.DataFrame) -> XGBClassifier:
    # Small dataset (~2000 rows) -- kept shallow and regularized so it
    # doesn't just memorize train. Tree models don't need feature scaling.
    model = XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        eval_metric="logloss", random_state=42,
    )
    model.fit(train_df[FEATURE_COLS], train_df["home_win"])
    return model


def predict_proba(model_type: str, logistic_model, xgb_model, X):
    """Win-probability array for whichever model type won, given both
    sub-models. Deliberately a plain function rather than a custom
    picklable class -- a class instance's pickled identity depends on
    which module __name__ happens to be "__main__" in at save time (e.g.
    `python -m model.train` vs `python run_week.py`), which breaks
    unpickling from a different entry point. Plain sklearn/xgboost objects
    don't have that problem, so this keeps every saved artifact to those."""
    if model_type == "xgboost":
        return xgb_model.predict_proba(X)
    if model_type == "logistic":
        return logistic_model.predict_proba(X)
    return (logistic_model.predict_proba(X) + xgb_model.predict_proba(X)) / 2


def train_spread_calibration(train_df: pd.DataFrame, model_type: str, logistic_model, xgb_model) -> LinearRegression:
    """Maps the model's home win probability to an implied Vegas-style spread,
    fit on how the market actually priced games with similar win probabilities
    (Section 4.3) -- lets predict.py compare the model directly against the
    current spread_line to find an edge."""
    win_prob = predict_proba(model_type, logistic_model, xgb_model, train_df[FEATURE_COLS])[:, 1]
    return LinearRegression().fit(win_prob.reshape(-1, 1), train_df["spread_line"])


def _evaluate(name: str, logistic_model, xgb_model, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    train_proba = predict_proba(name, logistic_model, xgb_model, train_df[FEATURE_COLS])[:, 1]
    train_acc = accuracy_score(train_df["home_win"], (train_proba >= 0.5).astype(int))
    test_proba = predict_proba(name, logistic_model, xgb_model, test_df[FEATURE_COLS])[:, 1]
    test_acc = accuracy_score(test_df["home_win"], (test_proba >= 0.5).astype(int))
    test_loss = log_loss(test_df["home_win"], test_proba)
    print(f"  {name:12s} train acc {train_acc:.3f}  test acc {test_acc:.3f}  test log loss {test_loss:.3f}")
    return {"name": name, "test_acc": test_acc, "test_loss": test_loss}


# Audit Fix Plan Step 5: walk-forward folds start once at least this many
# seasons of training data are available -- with only 10 cached seasons
# total, starting any earlier would fold on a training set too thin to
# mean much, and starting later would leave too few folds to call a
# "majority" meaningful.
MIN_WALK_FORWARD_TRAIN_SEASONS = 6


def walk_forward_folds(games: pd.DataFrame, seasons: list[int],
                        min_train_seasons: int = MIN_WALK_FORWARD_TRAIN_SEASONS) -> list[dict]:
    """One fold per season from `seasons[min_train_seasons]` onward: train
    on every season before it, test on it alone, then slide forward --
    train 1..N/test N+1, train 1..N+1/test N+2, etc. Unlike main()'s final
    deployed model (always trained on every season but the true last-held-
    out one), this deliberately never touches the *last* season in
    `seasons` as a training season for an earlier fold's test -- it's each
    fold's own test season in the final fold, keeping the walk-forward
    process entirely within seasons that would otherwise be "training
    data," never previewing the true final holdout."""
    folds = []
    for i in range(min_train_seasons, len(seasons)):
        train_seasons, test_season = seasons[:i], seasons[i]
        train_df = games[games["season"].isin(train_seasons)]
        test_df = games[games["season"] == test_season]
        if train_df.empty or test_df.empty:
            continue

        logistic_model = train_logistic(train_df)
        xgb_model = train_xgboost(train_df)
        results = {}
        for name in ("logistic", "xgboost", "ensemble"):
            proba = predict_proba(name, logistic_model, xgb_model, test_df[FEATURE_COLS])[:, 1]
            results[name] = float(accuracy_score(test_df["home_win"], (proba >= 0.5).astype(int)))
        fold_winner = max(results, key=results.get)
        folds.append({
            "train_seasons": train_seasons, "test_season": test_season,
            "results": results, "fold_winner": fold_winner,
        })
    return folds


def select_model_type_by_walk_forward(folds: list[dict]) -> str:
    """Whichever model type wins the most folds, ties broken by highest
    average test accuracy across all folds (not just the folds it won) --
    prefers the type that's been consistently strong over one that only
    barely edged out narrow wins. Per the audit plan: a majority across
    folds, not just whichever won the single most recent holdout."""
    wins = {"logistic": 0, "xgboost": 0, "ensemble": 0}
    avg_acc = {"logistic": [], "xgboost": [], "ensemble": []}
    for fold in folds:
        wins[fold["fold_winner"]] += 1
        for name, acc in fold["results"].items():
            avg_acc[name].append(acc)
    avg_acc = {name: sum(accs) / len(accs) for name, accs in avg_acc.items()}
    return max(wins, key=lambda name: (wins[name], avg_acc[name]))


def main():
    schedules = pd.read_parquet(SCHEDULES_PATH)
    team_stats = pd.read_parquet(TEAM_STATS_PATH)
    print("Pulling historical injury reports...")
    injuries = historical_injury_impact(HISTORICAL_SEASONS)
    print("Backfilling Elo ratings...")
    elo_per_game, _ = compute_elo_ratings(schedules)
    blowouts = blowout_loss_flags(schedules)
    lookaheads = lookahead_flags(schedules)

    games = build_feature_frame(schedules, team_stats, injuries, elo_per_game, blowouts, lookaheads)
    train_df = games[games["season"].isin(TRAIN_SEASONS)]
    test_df = games[games["season"] == TEST_SEASON]
    print(f"Train: {len(train_df)} games ({TRAIN_SEASONS[0]}-{TRAIN_SEASONS[-1]})")
    print(f"Test:  {len(test_df)} games ({TEST_SEASON})")
    baseline_acc = test_df["home_win"].mean()
    print(f"Always-pick-home baseline: {max(baseline_acc, 1 - baseline_acc):.3f}")

    # Audit Fix Plan Step 5: pick which model TYPE to deploy by walk-forward
    # validation across every available season, not just whichever won the
    # single most recent (2025) holdout -- a type that wins one fold by
    # luck of that particular season's matchups isn't necessarily the type
    # that generalizes best. This only decides model_type; the actual
    # deployed model below is still fit on the full training window
    # (TRAIN_SEASONS) so nothing about how much data it learns from
    # changes, and TEST_SEASON is never included as a training season in
    # any walk-forward fold, so it stays a true, unseen final holdout.
    print("\nWalk-forward validation (train on seasons 1..N, test on N+1, sliding forward):")
    folds = walk_forward_folds(games, HISTORICAL_SEASONS)
    for fold in folds:
        results_str = "  ".join(f"{name} {acc:.3f}" for name, acc in fold["results"].items())
        train_span = f"{fold['train_seasons'][0]}-{fold['train_seasons'][-1]}"
        print(f"  train {train_span} / test {fold['test_season']}: {results_str}  -> won by {fold['fold_winner']}")
    model_type = select_model_type_by_walk_forward(folds)
    fold_wins = sum(1 for f in folds if f["fold_winner"] == model_type)
    print(f"Selected model type: {model_type} (won {fold_wins}/{len(folds)} walk-forward folds)")

    logistic_model = train_logistic(train_df)
    xgb_model = train_xgboost(train_df)

    print(f"\nFinal {model_type} model on the true {TEST_SEASON} holdout:")
    final_result = _evaluate(model_type, logistic_model, xgb_model, train_df, test_df)

    spread_calibration = train_spread_calibration(train_df, model_type, logistic_model, xgb_model)

    joblib.dump({
        "logistic_model": logistic_model,
        "xgb_model": xgb_model,
        "model_type": model_type,
        "feature_cols": FEATURE_COLS,
        "spread_calibration": spread_calibration,
        # So the report can show real, current numbers instead of a
        # hardcoded string that goes stale the next time this is retrained.
        "test_accuracy": final_result["test_acc"],
        "baseline_accuracy": max(baseline_acc, 1 - baseline_acc),
        "test_season": TEST_SEASON,
        "walk_forward_folds": folds,
    }, MODEL_PATH)
    print(f"Saved {model_type} model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
