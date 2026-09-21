# run_week.py's result logging: pandas-3 grading crash, and never logging a game
# after it has kicked off.
import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

import model.prediction_log as prediction_log
import run_week
from model.prediction_log import game_has_kicked_off, log_predictions

UTC = dt.timezone.utc
BEFORE = dt.datetime(2026, 9, 20, 16, 30, tzinfo=UTC)   # 12:30 ET Sunday
AFTER = dt.datetime(2026, 9, 20, 17, 5, tzinfo=UTC)     # 1:05 PM ET -- the 1pm game is under way


def _schedule(scores=None):
    """Week 1: SEA @ NE was played; Week 2: nothing played unless `scores` says so."""
    rows = [{"season": 2026, "week": 1, "home_team": "NE", "away_team": "SEA", "home_score": 17.0, "away_score": 20.0}]
    rows += [{"season": 2026, "week": 2, "home_team": h, "away_team": a,
              "home_score": (scores or {}).get((h, a), (np.nan, np.nan))[0],
              "away_score": (scores or {}).get((h, a), (np.nan, np.nan))[1]}
             for h, a in [("BUF", "DET"), ("ATL", "CAR")]]
    return pd.DataFrame(rows)


def _pending_csv(path, rows):
    cols = run_week.LOG_COLS
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


def _week1_pending_row():
    return {"season": 2026, "week": 1, "away_team": "SEA", "home_team": "NE", "predicted_winner": "NE",
            "home_win_prob": 0.7, "vegas_spread": 3.0, "model_spread": 6.0, "edge": 3.0}


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setattr(run_week, "LOG_PATH", str(tmp_path / "season_results.csv"))
    monkeypatch.setattr(run_week, "PROPS_LOG_PATH", str(tmp_path / "props_results.csv"))
    monkeypatch.setattr(run_week.nfl, "import_schedules", lambda seasons: _schedule())
    return tmp_path


# ---- pandas 3: grading a pending row -------------------------------------

def test_grading_a_pending_row_no_longer_crashes_on_an_all_empty_actual_winner_column(paths):
    # exactly the state a real log is in after the first run: every "actual_*" column empty,
    # which read_csv types float64 -- assigning "SEA" used to raise TypeError.
    _pending_csv(run_week.LOG_PATH, [_week1_pending_row()])
    assert pd.read_csv(run_week.LOG_PATH)["actual_winner"].dtype == "float64"  # the trap is real

    graded = run_week._grade_pending(run_week._load_log())

    assert graded.at[0, "actual_winner"] == "SEA"
    assert graded.at[0, "correct"] == False  # noqa: E712  (NE predicted, SEA won)
    assert graded.at[0, "actual_home_score"] == 17.0 and graded.at[0, "actual_away_score"] == 20.0


def test_a_tie_grades_without_crashing(paths, monkeypatch):
    monkeypatch.setattr(run_week.nfl, "import_schedules",
                        lambda seasons: _schedule().assign(home_score=[20.0, np.nan, np.nan],
                                                           away_score=[20.0, np.nan, np.nan]))
    _pending_csv(run_week.LOG_PATH, [_week1_pending_row()])
    graded = run_week._grade_pending(run_week._load_log())
    assert graded.at[0, "actual_winner"] == "TIE" and pd.isna(graded.at[0, "correct"])


def test_props_log_loads_with_assignable_dtypes(paths):
    pd.DataFrame([{c: np.nan for c in run_week.PROPS_LOG_COLS}]).to_csv(run_week.PROPS_LOG_PATH, index=False)
    df = run_week._load_props_log()
    for col in ("actual_over", "model_correct", "market_correct", "model_beat_market"):
        df.loc[0, col] = True                      # would raise TypeError on a float64 column
    df.loc[0, "actual_value"] = 42.0
    assert df.at[0, "actual_over"] == True  # noqa: E712


# ---- never log after kickoff ---------------------------------------------

def _predictions():
    base = {"season": 2026, "week": 2, "home_win_prob": 0.6, "spread_line": 3.0, "implied_spread": 4.0,
            "edge": 1.0, "total_line": 44.0, "lines_source": "nflverse", "top_factors": [("market_spread", 0.2)]}
    return pd.DataFrame([
        {**base, "game_id": "2026_02_CAR_ATL", "home_team": "ATL", "away_team": "CAR",
         "gameday": "2026-09-20", "gametime": "13:00"},            # 1pm ET Sunday
        {**base, "game_id": "2026_02_DET_BUF", "home_team": "BUF", "away_team": "DET",
         "gameday": "2026-09-20", "gametime": "20:20"},            # SNF -- still ahead at AFTER
    ])


def test_game_has_kicked_off():
    assert not game_has_kicked_off("2026-09-20", "13:00", BEFORE)     # 13:00 ET == 17:00Z
    assert game_has_kicked_off("2026-09-20", "13:00", AFTER)
    assert not game_has_kicked_off("2026-09-20", "20:20", AFTER)
    assert not game_has_kicked_off("2026-09-20", None, AFTER)         # unconfirmed time: not "started"
    assert not game_has_kicked_off(None, "13:00", AFTER)


def test_log_week_keeps_a_started_games_pre_kickoff_row_and_adds_nothing_new_for_it(paths):
    pre = {"season": 2026, "week": 2, "away_team": "CAR", "home_team": "ATL", "predicted_winner": "CAR",
           "home_win_prob": 0.39, "vegas_spread": -2.5, "model_spread": -2.5, "edge": 0.0}
    stale_other = {**pre, "away_team": "DET", "home_team": "BUF", "predicted_winner": "BUF", "home_win_prob": 0.5}
    _pending_csv(run_week.LOG_PATH, [pre, stale_other])

    log, _ = run_week.log_week(_predictions().assign(home_win_prob=0.9), 2, 2026, now=AFTER)

    week2 = log[log["week"] == 2].set_index("home_team")
    assert week2.at["ATL", "home_win_prob"] == 0.39      # started: the pre-kickoff row survives, not the hindsight 0.9
    assert week2.at["BUF", "home_win_prob"] == 0.9       # not started: refreshed as usual
    assert len(week2) == 2


def test_log_week_keeps_an_unstarted_games_earlier_row_when_this_run_has_no_prediction_for_it(paths):
    # a line got pulled / a team-name mapping missed: this run produced NaN for a game that has NOT
    # started. Its earlier pre-kickoff row must survive -- it can't be recreated once the game starts.
    earlier = {"season": 2026, "week": 2, "away_team": "DET", "home_team": "BUF", "predicted_winner": "BUF",
               "home_win_prob": 0.62, "vegas_spread": 5.5, "model_spread": 6.0, "edge": 0.5}
    _pending_csv(run_week.LOG_PATH, [earlier])
    preds = _predictions()
    preds.loc[preds["home_team"] == "BUF", "home_win_prob"] = np.nan      # BUF game: no prediction this run

    log, _ = run_week.log_week(preds, 2, 2026, now=BEFORE)                # nothing has started

    week2 = log[log["week"] == 2].set_index("home_team")
    assert week2.at["BUF", "home_win_prob"] == 0.62                       # earlier row untouched
    assert week2.at["ATL", "home_win_prob"] == 0.6                        # the game that DID get a prediction is logged


def test_a_game_with_a_final_score_counts_as_started_even_without_a_kickoff_time(paths):
    preds = _predictions().assign(gametime=None)                          # kickoff times unconfirmed
    preds.loc[preds["home_team"] == "ATL", ["home_score", "away_score"]] = [17.0, 20.0]   # ATL game is over
    log, _ = run_week.log_week(preds, 2, 2026, now=BEFORE)
    assert list(log["home_team"]) == ["BUF"]                              # only the unplayed game is logged


def test_log_week_writes_nothing_for_a_started_game_that_was_never_logged(paths):
    log, _ = run_week.log_week(_predictions(), 2, 2026, now=AFTER)
    assert list(log["home_team"]) == ["BUF"]


def test_log_props_week_skips_and_preserves_started_teams(paths, monkeypatch):
    monkeypatch.setattr(run_week, "_grade_pending_props", lambda df: df)   # grading (real pbp) isn't under test
    def prop(player, team, prob):
        return {"season": 2026, "week": 2, "player": player, "player_id": player, "team": team, "opponent": "X",
                "stat": "pass_yards", "line": 250.5, "market_over_prob": 0.5, "model_over_prob": prob,
                "edge": 0.0, "has_line": True}
    pd.DataFrame([{k: v for k, v in prop("A", "ATL", 0.40).items() if k != "has_line"}]).reindex(
        columns=run_week.PROPS_LOG_COLS).to_csv(run_week.PROPS_LOG_PATH, index=False)

    props = pd.DataFrame([prop("A", "ATL", 0.99), prop("B", "BUF", 0.7), prop("C", "ATL", 0.8)])
    log, _ = run_week.log_props_week(props, 2, 2026, started={"ATL"})

    by_player = log.set_index("player")
    assert by_player.at["A", "model_over_prob"] == 0.40     # started team's pre-kickoff row kept
    assert "C" not in by_player.index                       # nothing new for a started team
    assert by_player.at["B", "model_over_prob"] == 0.7      # not started: logged


def test_log_props_week_does_not_wipe_the_log_when_no_prop_has_a_market_line(paths, monkeypatch):
    # Odds API down -> every prop has has_line=False -> nothing to re-log. The week's earlier,
    # valid pre-kickoff rows must survive (this used to empty the props log).
    monkeypatch.setattr(run_week, "_grade_pending_props", lambda df: df)
    row = {"season": 2026, "week": 2, "player": "A", "player_id": "A", "team": "BUF", "opponent": "DET",
           "stat": "pass_yards", "line": 250.5, "market_over_prob": 0.5, "model_over_prob": 0.7, "edge": 0.0}
    pd.DataFrame([row]).reindex(columns=run_week.PROPS_LOG_COLS).to_csv(run_week.PROPS_LOG_PATH, index=False)
    props = pd.DataFrame([{**row, "has_line": False}])

    log, _ = run_week.log_props_week(props, 2, 2026, started=set())

    assert list(log["player"]) == ["A"]


def test_grading_failure_does_not_block_logging_new_predictions(paths, monkeypatch, capsys):
    _pending_csv(run_week.LOG_PATH, [_week1_pending_row()])

    def boom(df):
        raise ConnectionResetError("connection reset by peer")

    monkeypatch.setattr(run_week, "_grade_pending", boom)
    log, newly = run_week.log_week(_predictions(), 2, 2026, now=BEFORE)

    assert sorted(log[log["week"] == 2]["home_team"]) == ["ATL", "BUF"]   # this run's picks are logged
    assert log[log["week"] == 1]["actual_winner"].isna().all()             # week 1 simply stays pending
    assert newly.empty
    assert "couldn't grade earlier weeks" in capsys.readouterr().out


def test_results_csv_is_written_atomically(paths):
    run_week._write_csv(pd.DataFrame({"a": [1]}), run_week.LOG_PATH)
    assert pd.read_csv(run_week.LOG_PATH)["a"].tolist() == [1]
    assert not __import__("os").path.exists(run_week.LOG_PATH + ".tmp")   # temp file renamed away


def test_ledger_skips_started_games_and_records_the_lines_source(tmp_path, monkeypatch):
    ledger = tmp_path / "predictions.jsonl"
    monkeypatch.setattr(prediction_log, "PREDICTIONS_LOG_PATH", str(ledger))
    model_file = tmp_path / "model.joblib"
    model_file.write_bytes(b"frozen-model-bytes")

    path, n = log_predictions(_predictions(), 2, 2026, model_path=str(model_file), now=AFTER)

    records = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert n == 1 and [r["game_id"] for r in records] == ["2026_02_DET_BUF"]
    assert records[0]["market_lines_source"] == "nflverse"
    assert records[0]["logged_at_utc"] < records[0]["kickoff_utc"]   # logged before its own kickoff


def test_ledger_skips_a_game_with_a_score_even_if_its_kickoff_time_is_missing(tmp_path, monkeypatch):
    ledger = tmp_path / "predictions.jsonl"
    monkeypatch.setattr(prediction_log, "PREDICTIONS_LOG_PATH", str(ledger))
    model_file = tmp_path / "model.joblib"
    model_file.write_bytes(b"m")
    preds = _predictions().assign(gametime=None)
    preds.loc[preds["home_team"] == "ATL", ["home_score", "away_score"]] = [17.0, 20.0]

    _, n = log_predictions(preds, 2, 2026, model_path=str(model_file), now=BEFORE)

    records = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert n == 1 and [r["game_id"] for r in records] == ["2026_02_DET_BUF"]


def test_kickoff_parsing_tolerates_a_seconds_field_and_rejects_garbage():
    from model.prediction_log import _kickoff_utc
    assert _kickoff_utc("2026-09-20", "13:00:00") == _kickoff_utc("2026-09-20", "13:00")
    assert _kickoff_utc("2026-09-20", "sometime") is None
