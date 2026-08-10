# tests for the preseason/roster-turnover confidence caveat (Combined Build Plan Part 2)
"""model/predict.py's fallback_confidence_caveat() is pure decision logic
over two already-computed dicts (team_roster_turnover()'s per-team churn,
team_fallback_status()'s per-team games-played/thin-sample flag) -- both
of those upstream functions need live nfl_data_py calls, but the actual
AND-gate this test covers doesn't, so it's exercised directly with
synthetic inputs instead of mocking a network fetch just to test a
four-line conditional.
"""

from model.predict import TURNOVER_CAVEAT_THRESHOLD, fallback_confidence_caveat
from report.cards import fallback_caveat_text


def _turnover(pct, departed=7, total=20):
    return {"SEA": {"turnover_pct": pct, "departed": departed, "total_starters": total}}


def _status(thin, games=1):
    return {"SEA": {"games_played": games, "thin_sample": thin}}


def test_caveat_fires_only_when_both_thin_sample_and_high_turnover():
    result = fallback_confidence_caveat(
        "SEA", 2025, _turnover(TURNOVER_CAVEAT_THRESHOLD + 0.05), _status(thin=True))
    assert result is not None
    assert result["prior_season"] == 2025


def test_no_caveat_when_sample_is_not_thin():
    """High turnover on a team that already has enough real current-
    season games shouldn't caveat -- the model is already using real
    current data, not the prior-season fallback this caveat is about."""
    result = fallback_confidence_caveat(
        "SEA", 2025, _turnover(TURNOVER_CAVEAT_THRESHOLD + 0.05), _status(thin=False, games=6))
    assert result is None


def test_no_caveat_when_turnover_is_normal():
    """Thin sample alone isn't enough -- normal offseason churn (15-25%,
    per qa/validate_rosters.py's own documented range) on a team with no
    games yet shouldn't caveat every single preseason team."""
    result = fallback_confidence_caveat("SEA", 2025, _turnover(0.20), _status(thin=True))
    assert result is None


def test_no_caveat_for_team_missing_from_either_dict():
    """A team absent from turnover (no recorded prior-season starters)
    or fallback_status (shouldn't happen, but defensively) doesn't
    crash and doesn't caveat."""
    assert fallback_confidence_caveat("KC", 2025, _turnover(0.9), _status(thin=True)) is None
    assert fallback_confidence_caveat("SEA", 2025, {}, _status(thin=True)) is None
    assert fallback_confidence_caveat("SEA", 2025, _turnover(0.9), {}) is None


def test_caveat_boundary_exactly_at_threshold_fires():
    """Matches the source's `>=` comparison, not `>`."""
    result = fallback_confidence_caveat(
        "SEA", 2025, _turnover(TURNOVER_CAVEAT_THRESHOLD), _status(thin=True))
    assert result is not None


def test_fallback_caveat_text_renders_full_name_and_real_numbers():
    caveat = {"prior_season": 2025, "departed": 7, "total_starters": 20, "turnover_pct": 0.35}
    text = fallback_caveat_text(caveat, "Seattle Seahawks")
    assert "2025" in text
    assert "Seattle Seahawks" in text
    assert "7/20" in text


def test_fallback_caveat_text_none_when_no_caveat():
    assert fallback_caveat_text(None, "Seattle Seahawks") is None
