# Week 1 Audit & Tuning Plan Phase 6: append-only prediction ledger
"""An unlogged model is a toy; a logged one is a real track record. This
writes one JSONL line per graded game to logs/predictions.jsonl, opened
in 'a' (append) mode ONLY -- the file is never read back and rewritten,
so nothing already on disk can ever be edited or deleted by a later
call. That's what "never allow overwriting an existing prediction
record" means here in practice: the write path itself is incapable of
it, not just a convention this module promises to follow.

Multiple records for the same game_id across separate calls (e.g. a
Wednesday run vs. a Friday run, if the line has moved) are expected,
not a bug -- "market spread and total at time of prediction" only means
something if the ledger keeps every snapshot instead of collapsing to
whichever run happened most recently. grade.py (project root) is what
joins these snapshots to final scores once each game is over.
"""

import datetime as dt
import hashlib
import json
import os
from zoneinfo import ZoneInfo

import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
PREDICTIONS_LOG_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "logs", "predictions.jsonl")
)

# nflverse's schedule `gametime` is documented Eastern Time regardless of
# the game's actual venue -- verified against 2025's Sao Paulo game
# (2025_01_KC_LAC, location="Neutral"), which still carries a plain
# ET-slot time (20:00) exactly like every domestic game, not a
# Brazil-local kickoff hour. Every kickoff below converts through this
# zone, never a venue-local one.
_SCHEDULE_TZ = ZoneInfo("America/New_York")


def model_version_hash(model_path: str = MODEL_PATH, length: int = 12) -> str:
    """sha256 of the deployed model.joblib's raw bytes, truncated for a
    readable but still collision-safe identifier -- a prediction logged
    against one hash is provably tied to one specific trained artifact,
    not just "whatever main() happened to save most recently.\""""
    with open(model_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    return digest[:length]


def _kickoff_utc(gameday, gametime) -> str | None:
    """None if either half of the schedule's kickoff slot is missing --
    a flex-scheduled or not-yet-time-confirmed game still has a real
    spread_line and home_win_prob (model/predict.py:314 only requires a
    posted spread to score a game), so this must degrade gracefully
    rather than raise and take down the whole logging call over one
    non-essential field."""
    if pd.isna(gameday) or pd.isna(gametime):
        return None
    for time_format in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):   # tolerate a seconds field
        try:
            naive = dt.datetime.strptime(f"{gameday} {gametime}", time_format)
            break
        except ValueError:
            continue
    else:
        return None   # unparseable: same "no confirmed kickoff" handling as a missing time
    localized = naive.replace(tzinfo=_SCHEDULE_TZ)
    return localized.astimezone(dt.timezone.utc).isoformat()


def game_has_kicked_off(gameday, gametime, now: dt.datetime | None = None, home_score=None) -> bool:
    """True once a game's scheduled kickoff has passed, or as soon as it has a final
    score -- a game with no confirmed kickoff time (which otherwise counts as NOT
    started, same "degrade gracefully" stance as _kickoff_utc) is certainly over once
    a score exists. Anything predicted after this is True is a hindsight number, not
    a prediction, and must never be logged as one."""
    if home_score is not None and pd.notna(home_score):
        return True
    kickoff = _kickoff_utc(gameday, gametime)
    if kickoff is None:
        return False
    now = now or dt.datetime.now(dt.timezone.utc)
    return dt.datetime.fromisoformat(kickoff) <= now


def pre_game_picks(path: str | None = None) -> dict[str, str]:
    """game_id -> predicted winner, from the EARLIEST ledger record for that game
    that was logged before its kickoff. A game with no such record has no valid
    pre-game pick on file (e.g. it was already played the first time the week was
    scored), so the report must not present a later, hindsight number as one."""
    path = path or PREDICTIONS_LOG_PATH
    if not os.path.exists(path):
        return {}
    earliest: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("kickoff_utc") or r["logged_at_utc"] >= r["kickoff_utc"]:
                continue
            if r["game_id"] not in earliest or r["logged_at_utc"] < earliest[r["game_id"]]["logged_at_utc"]:
                earliest[r["game_id"]] = r
    return {gid: (r["home_team"] if r["home_win_prob"] >= 0.5 else r["away_team"]) for gid, r in earliest.items()}


def log_predictions(predictions: pd.DataFrame, week: int, season: int,
                     model_path: str = MODEL_PATH, now: dt.datetime | None = None) -> tuple[str, int]:
    """Appends one record per game that actually got a prediction (a game
    with no history for one of its teams has home_win_prob=None from
    score_week() and nothing gradeable to log). A game that has already kicked
    off is skipped: re-running a week mid-week (e.g. Sunday morning, after
    Thursday night's game) would otherwise log a hindsight "prediction" that
    grade.py later grades as if it had been made pre-game. Returns (log path,
    count of records written this call)."""
    version = model_version_hash(model_path)
    # one instant for both the "has it kicked off?" decision and the timestamp
    # written on every record, so a record can never say it was logged after a
    # kickoff it was allowed to be logged before (or the reverse)
    now = now or dt.datetime.now(dt.timezone.utc)
    logged_at = now.isoformat()

    os.makedirs(os.path.dirname(PREDICTIONS_LOG_PATH), exist_ok=True)
    n_written = 0
    with open(PREDICTIONS_LOG_PATH, "a") as f:
        for _, game in predictions.iterrows():
            if pd.isna(game.get("home_win_prob")):
                continue
            if game_has_kicked_off(game.get("gameday"), game.get("gametime"), now, game.get("home_score")):
                print(f"Not logging {game['away_team']} @ {game['home_team']}: already kicked off.")
                continue
            kickoff_utc = _kickoff_utc(game.get("gameday"), game.get("gametime"))
            if kickoff_utc is None:
                print(f"Warning: no confirmed kickoff time for {game['game_id']}; "
                      f"logging the prediction with kickoff_utc=null rather than skipping it.")
            record = {
                "logged_at_utc": logged_at,
                "game_id": game["game_id"],
                "season": season,
                "week": week,
                "kickoff_utc": kickoff_utc,
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "home_win_prob": float(game["home_win_prob"]),
                "predicted_spread": float(game["implied_spread"]),
                "market_spread": float(game["spread_line"]) if pd.notna(game.get("spread_line")) else None,
                "market_total": float(game["total_line"]) if pd.notna(game.get("total_line")) else None,
                "market_lines_source": game.get("lines_source") if pd.notna(game.get("lines_source")) else None,
                "model_version": version,
                "top_factors": [[name, float(value)] for name, value in (game.get("top_factors") or [])],
            }
            f.write(json.dumps(record) + "\n")
            n_written += 1
    return PREDICTIONS_LOG_PATH, n_written
