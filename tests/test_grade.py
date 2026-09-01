# tests for grade.py's pure grading arithmetic (Week 1 Audit & Tuning Plan Phase 6)
from grade import grade_one


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
