# Week 1 Audit & Tuning Plan Phase 6: grades the append-only prediction ledger
"""Joins logs/predictions.jsonl (model/prediction_log.py) to final scores
once each game is over, appending one grading record per (game_id,
logged_at_utc) prediction snapshot to logs/predictions_graded.jsonl.
Like the ledger it reads, this only ever appends -- an already-graded
snapshot is skipped (tracked by re-reading what's already in the graded
file), never re-derived or rewritten. Safe to run repeatedly, e.g. daily
alongside run_week.py.

This is a separate, stricter ledger from logs/season_results.csv
(run_week.py's own log_week()/`_grade_pending()`) -- that CSV is the
report/track-record page's mutable running tally (re-running a week
intentionally replaces its rows) and pre-dates this plan. This file is
the portfolio-grade audit trail Phase 6 asks for: immutable, append-
only, and tied to the specific model_version hash that produced each
prediction.
"""

import datetime as dt
import json
import os

import nfl_data_py as nfl
import pandas as pd

from model.prediction_log import PREDICTIONS_LOG_PATH

GRADED_LOG_PATH = os.path.join(os.path.dirname(PREDICTIONS_LOG_PATH), "predictions_graded.jsonl")


def _load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def grade_one(prediction: dict, home_score: float, away_score: float, graded_at_utc: str) -> dict:
    """Pure grading arithmetic for a single already-played game -- kept
    separate from grade()'s file I/O and the nfl_data_py schedule lookup
    so this logic (straight-up + ATS correctness, push exclusion) is
    unit-testable without a network call."""
    if home_score == away_score:
        # A real NFL tie -- neither team "won," so there's nothing to grade
        # the straight-up pick right or wrong against. Distinct from a push
        # in the ATS block below, which stays unaffected: ATS grades margin
        # against the spread, not who won.
        actual_winner, straight_up_correct = "TIE", None
    else:
        actual_winner = prediction["home_team"] if home_score > away_score else prediction["away_team"]
        predicted_winner = prediction["home_team"] if prediction["home_win_prob"] >= 0.5 else prediction["away_team"]
        straight_up_correct = actual_winner == predicted_winner

    ats_correct = None
    if prediction.get("market_spread") is not None:
        actual_margin = home_score - away_score
        if actual_margin != prediction["market_spread"]:  # exclude a push -- nothing to grade
            home_covered = actual_margin > prediction["market_spread"]
            model_picked_home_covers = prediction["predicted_spread"] > prediction["market_spread"]
            ats_correct = model_picked_home_covers == home_covered

    return {
        "game_id": prediction["game_id"], "logged_at_utc": prediction["logged_at_utc"],
        "graded_at_utc": graded_at_utc,
        "actual_home_score": home_score, "actual_away_score": away_score,
        "actual_winner": actual_winner,
        "straight_up_correct": straight_up_correct,
        "ats_correct": ats_correct,
    }


def grade() -> int:
    predictions = _load_jsonl(PREDICTIONS_LOG_PATH)
    if not predictions:
        return 0

    already_graded = {(g["game_id"], g["logged_at_utc"]) for g in _load_jsonl(GRADED_LOG_PATH)}
    pending = [p for p in predictions if (p["game_id"], p["logged_at_utc"]) not in already_graded]
    if not pending:
        return 0

    seasons = sorted({p["season"] for p in pending})
    schedules = nfl.import_schedules(seasons)
    graded_at = dt.datetime.now(dt.timezone.utc).isoformat()

    os.makedirs(os.path.dirname(GRADED_LOG_PATH), exist_ok=True)
    n_written = 0
    with open(GRADED_LOG_PATH, "a") as f:
        for p in pending:
            game = schedules[schedules["game_id"] == p["game_id"]]
            if game.empty or pd.isna(game.iloc[0]["home_score"]):
                continue  # not played yet
            home_score, away_score = float(game.iloc[0]["home_score"]), float(game.iloc[0]["away_score"])
            record = grade_one(p, home_score, away_score, graded_at)
            f.write(json.dumps(record) + "\n")
            n_written += 1
    return n_written


def main():
    n = grade()
    print(f"Graded {n} new prediction record(s) -> {GRADED_LOG_PATH}")


if __name__ == "__main__":
    main()
