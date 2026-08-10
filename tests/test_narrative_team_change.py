# tests for the team-change narrative candidate (Combined Build Plan Part 3)
"""report/narrative.py had no test coverage before this -- rather than
retroactively covering the whole pre-existing module, this covers just
the new team_change candidate/phraser this build adds, and confirms it
composes correctly with select_lead_narrative()'s existing candidate
competition (positional/streak/h2h), which the earlier commits already
exercised in production without a dedicated test file.
"""

from report.narrative import (
    TEAM_CHANGE_TURNOVER_NOTABLE, _phrase_team_change, _team_change_candidate, phrase_lead_narrative,
    select_lead_narrative,
)

COACH_CHANGE = {"prior_coach": "Coach A", "current_coach": "Coach B"}
HIGH_TURNOVER = {"secondary": {"turnover_pct": TEAM_CHANGE_TURNOVER_NOTABLE + 0.1, "departed": 5, "total": 8}}


def test_team_change_candidate_supports_winner_via_losers_instability():
    """The framing this whole feature depends on: it's the LOSER's team
    change that supports picking the winner, not the winner's own."""
    team_changes = {"NE": {"coach_change": COACH_CHANGE, "unit_turnover": HIGH_TURNOVER}}
    result = _team_change_candidate("SEA", "NE", team_changes)
    assert result is not None
    score, candidate = result
    assert candidate["team"] == "NE"
    assert score > 1.0  # coach change (1.0) + turnover severity on top


def test_team_change_candidate_none_for_the_winners_own_turnover():
    """The winner having a new coach/high turnover isn't a supporting
    reason for THEM winning -- it should never surface as a candidate
    just because the winner happens to be in team_changes."""
    team_changes = {"SEA": {"coach_change": COACH_CHANGE, "unit_turnover": HIGH_TURNOVER}}
    assert _team_change_candidate("SEA", "NE", team_changes) is None


def test_team_change_candidate_none_when_nothing_flagged():
    assert _team_change_candidate("SEA", "NE", {}) is None
    assert _team_change_candidate("SEA", "NE", {"NE": {"coach_change": None, "unit_turnover": {}}}) is None


def test_select_lead_narrative_defaults_team_changes_to_empty():
    """Every pre-Part-3 call site omits the new team_changes argument --
    must not error, must behave as if nothing was flagged."""
    result = select_lead_narrative("SEA", "NE", [], [None, None], [None, None], None, None)
    assert result is None


def test_select_lead_narrative_can_pick_team_change_as_the_lead():
    """When team-change is the ONLY candidate that clears its bar, it
    wins by default -- confirms it's actually wired into the candidate
    list, not just computable in isolation."""
    team_changes = {"NE": {"coach_change": COACH_CHANGE, "unit_turnover": HIGH_TURNOVER}}
    result = select_lead_narrative("SEA", "NE", [], [None, None], [None, None], None, None, team_changes)
    assert result is not None
    assert result["type"] == "team_change"
    assert result["team"] == "NE"


def test_phrase_team_change_mentions_coach_and_worst_unit():
    candidate = {"type": "team_change", "team": "NE", "coach_change": COACH_CHANGE, "unit_turnover": HIGH_TURNOVER}
    text = phrase_lead_narrative(candidate, "SEA", "Coach Winner", lambda abbr: {"NE": "New England Patriots"}[abbr])
    assert "New England Patriots" in text
    assert "Coach B" in text
    assert "secondary" in text
    assert "5 of 8" in text


def test_phrase_team_change_handles_coach_change_only():
    """A team with a new coach but no notably-turned-over unit should
    still phrase cleanly -- no dangling "and" or empty second clause
    referencing a unit that was never actually flagged."""
    candidate = {"type": "team_change", "team": "NE", "coach_change": COACH_CHANGE, "unit_turnover": {}}
    text = _phrase_team_change(candidate, lambda abbr: "New England Patriots")
    assert "Coach B" in text
    assert "turnover at" not in text
