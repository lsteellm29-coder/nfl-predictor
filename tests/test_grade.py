# tests for grade.py's pure grading arithmetic (Week 1 Audit & Tuning Plan Phase 6)
import json

import pandas as pd

import grade as grade_module
from grade import grade_one, logged_before_kickoff


def _prediction(**overrides):
    base = {
        "game_id": "2026_01_NE_SEA", "logged_at_utc": "2026-09-08T12:00:00+00:00",
        "home_team": "SEA", "away_team": "NE",
        "home_win_prob": 0.6, "predicted_spread": 3.0, "market_spread": 1.0,
    }
    base.update(overrides)
    return base


def test_grade_one_straight_up_correct_when_favored_team_wins():
    result = grade_one(_prediction(home_win_prob=0.6), home_score=24, away_score=17, graded_at_utc="t")
    assert result["actual_winner"] == "SEA"
    assert result["straight_up_correct"] is True


def test_grade_one_straight_up_wrong_when_underdog_wins():
    # model favored home (0.6) but away team actually won
    result = grade_one(_prediction(home_win_prob=0.6), home_score=17, away_score=24, graded_at_utc="t")
    assert result["actual_winner"] == "NE"
    assert result["straight_up_correct"] is False


def test_grade_one_ats_correct_when_model_and_market_agree_on_cover():
    # market_spread=1.0, predicted_spread=3.0 -> model picks "home covers".
    # actual margin 24-17=7 > 1.0 -> home did cover -> ATS correct.
    result = grade_one(_prediction(), home_score=24, away_score=17, graded_at_utc="t")
    assert result["ats_correct"] is True


def test_grade_one_ats_wrong_when_home_falls_short_of_the_market_number():
    # market_spread=3.0, predicted_spread=6.0 -> model picks "home covers"
    # (6.0 > 3.0). Actual margin 24-23=1, which is < 3.0 -- home did NOT
    # cover -- so the model's "covers" pick is wrong.
    result = grade_one(_prediction(market_spread=3.0, predicted_spread=6.0),
                        home_score=24, away_score=23, graded_at_utc="t")
    assert result["ats_correct"] is False


def test_grade_one_excludes_a_push_from_ats_grading():
    # actual margin exactly equals market_spread -> a push, nothing to grade
    result = grade_one(_prediction(market_spread=7.0), home_score=24, away_score=17, graded_at_utc="t")
    assert result["ats_correct"] is None


def test_grade_one_ats_none_when_no_market_spread_was_logged():
    result = grade_one(_prediction(market_spread=None), home_score=24, away_score=17, graded_at_utc="t")
    assert result["ats_correct"] is None


def test_grade_one_a_real_tie_is_not_graded_as_an_away_team_win():
    """Regression test: home_score > away_score being False does NOT mean
    the away team won -- a genuine NFL tie must be its own case, not
    silently graded as an away-team win."""
    result = grade_one(_prediction(), home_score=20, away_score=20, graded_at_utc="t")
    assert result["actual_winner"] == "TIE"
    assert result["straight_up_correct"] is None


def test_grade_one_tie_still_grades_ats_normally():
    # market_spread=1.0: actual margin 0 < 1.0 -> home did NOT cover.
    # predicted_spread=3.0 > market_spread=1.0 -> model picked "home covers".
    # Model's cover pick disagrees with the actual (no-cover) result -> wrong.
    result = grade_one(_prediction(), home_score=20, away_score=20, graded_at_utc="t")
    assert result["ats_correct"] is False


def test_grade_one_passes_through_identifying_fields():
    result = grade_one(_prediction(), home_score=24, away_score=17, graded_at_utc="2026-09-10T00:00:00+00:00")
    assert result["game_id"] == "2026_01_NE_SEA"
    assert result["logged_at_utc"] == "2026-09-08T12:00:00+00:00"
    assert result["graded_at_utc"] == "2026-09-10T00:00:00+00:00"
    assert result["actual_home_score"] == 24
    assert result["actual_away_score"] == 17


def test_logged_before_kickoff_eligibility():
    assert logged_before_kickoff(_prediction(logged_at_utc="2026-09-20T16:00:00+00:00",
                                             kickoff_utc="2026-09-20T17:00:00+00:00"))
    assert not logged_before_kickoff(_prediction(logged_at_utc="2026-09-20T18:00:00+00:00",
                                                 kickoff_utc="2026-09-20T17:00:00+00:00"))
    assert logged_before_kickoff(_prediction(kickoff_utc=None))            # can't be checked: stays eligible
    assert logged_before_kickoff(_prediction())                            # legacy record without the field


def test_grade_skips_hindsight_records_but_grades_real_ones(tmp_path, monkeypatch):
    ledger, graded = tmp_path / "predictions.jsonl", tmp_path / "predictions_graded.jsonl"
    good = _prediction(game_id="g_good", logged_at_utc="2026-09-20T16:00:00+00:00",
                       kickoff_utc="2026-09-20T17:00:00+00:00")
    late = _prediction(game_id="g_late", logged_at_utc="2026-09-20T18:00:00+00:00",
                       kickoff_utc="2026-09-20T17:00:00+00:00")
    ledger.write_text("".join(json.dumps(r) + "\n" for r in (good, late)))
    monkeypatch.setattr(grade_module, "PREDICTIONS_LOG_PATH", str(ledger))
    monkeypatch.setattr(grade_module, "GRADED_LOG_PATH", str(graded))
    schedule = pd.DataFrame([{"game_id": g, "home_score": 24.0, "away_score": 17.0} for g in ("g_good", "g_late")])
    monkeypatch.setattr(grade_module.nfl, "import_schedules", lambda seasons: schedule)
    for r in (good, late):
        r["season"] = 2026
    ledger.write_text("".join(json.dumps(r) + "\n" for r in (good, late)))

    assert grade_module.grade() == 1
    assert [json.loads(line)["game_id"] for line in graded.read_text().splitlines()] == ["g_good"]
