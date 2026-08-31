# tests for the explicit temporal-leakage tripwire (Week 1 Audit & Tuning Plan Phase 2)
"""data/leakage.py's assert_no_leakage() is the explicit, separately-
named guard at the week-boundary filters that are the one place a
future refactor could quietly weaken the no-leakage guarantee every
rolling-stat/Elo feature already has by construction (data/team_stats.py's
and data/opponent_adjust.py's .expanding().shift(1) pattern). These
tests cover the tripwire itself, not the underlying rolling-stat
construction (already verified directly by reading that code -- see
AUDIT.md's Phase 2 section).
"""

import pandas as pd
import pytest

from data.leakage import assert_no_leakage


def test_passes_when_every_row_is_strictly_before_the_target_week():
    df = pd.DataFrame({"team": ["SEA", "SEA", "NE"], "week": [1, 2, 1]})
    assert_no_leakage(df, week=3)  # should not raise


def test_raises_when_a_row_matches_the_target_week_exactly():
    """The most common real mistake this catches: a `<=` where a `<`
    belongs, which would include the game's own week."""
    df = pd.DataFrame({"team": ["SEA", "NE"], "week": [2, 3]})
    with pytest.raises(AssertionError, match="LEAKAGE"):
        assert_no_leakage(df, week=3)


def test_raises_when_a_row_is_from_a_future_week():
    df = pd.DataFrame({"team": ["SEA"], "week": [5]})
    with pytest.raises(AssertionError, match="LEAKAGE"):
        assert_no_leakage(df, week=3)


def test_empty_dataframe_never_raises():
    """No data at all (e.g. week 1 of a season with no prior games) is
    the honest "nothing to leak" case, not an error."""
    df = pd.DataFrame({"team": [], "week": []})
    assert_no_leakage(df, week=1)


def test_sentinel_upto_week_is_a_safe_no_op_for_a_complete_prior_season():
    """model/td_model.py's season_to_date()/recency_weighted_touch_share()
    both call this with upto_week=30 for a fully-completed fallback
    season -- no real season has a week 30, so this must never fire for
    that legitimate case."""
    df = pd.DataFrame({"team": ["SEA"] * 3, "week": [16, 17, 18]})
    assert_no_leakage(df, week=30)  # should not raise


def test_custom_week_column_name_is_respected():
    df = pd.DataFrame({"team": ["SEA"], "upto_week": [4]})
    with pytest.raises(AssertionError, match="LEAKAGE"):
        assert_no_leakage(df, week=4, week_col="upto_week")


def test_context_label_appears_in_the_error_message():
    """The context string is what makes a real failure immediately
    diagnosable -- confirms it actually gets included, not silently
    dropped."""
    df = pd.DataFrame({"team": ["SEA"], "week": [3]})
    with pytest.raises(AssertionError, match=r"\(_current_season_stats\)"):
        assert_no_leakage(df, week=3, context="_current_season_stats")
