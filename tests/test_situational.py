# tests for the neutral-site travel-penalty fix (Week 1 Audit & Tuning Plan Phase 3)
"""data/situational.py's away_travel_penalty() compares the two teams'
HOME market timezones -- which only means anything when one of them is
actually playing at home. For a neutral-site/international game (this
week's real 2026 LA @ SF game in Melbourne, confirmed via a live
nfl_data_py pull to be tagged location="Neutral"), neither team is in
their usual environment, so a timezone gap computed from their home
markets doesn't describe a real travel disadvantage for either side.
That specific matchup happens to not trigger the bug (LA and SF share a
timezone), but the gap is real and would misfire the next time an
international game pairs two teams from different U.S. timezones.
"""

from data.situational import away_travel_penalty


def test_cross_country_early_game_flags_without_neutral_site():
    """The existing, correct case: NE (home, ET) hosting LV (away, PT)
    for an early kickoff is a real West-Coast-team-flies-east spot."""
    assert away_travel_penalty("NE", "LV", "13:00") is True


def test_neutral_site_suppresses_the_penalty_even_with_a_real_timezone_gap():
    """The actual fix: the same NE/LV matchup, at a neutral site, must
    not flag -- neither team is playing in their home timezone, so
    comparing home-market timezones no longer describes anything real."""
    assert away_travel_penalty("NE", "LV", "13:00", neutral_site=True) is False


def test_same_timezone_matchup_never_flags_regardless_of_neutral_site():
    """This week's real matchup (LA/SF, both -3 offset) -- confirms the
    bug was genuinely dormant for this specific game, not accidentally
    fixed by something else."""
    assert away_travel_penalty("LA", "SF", "13:00") is False
    assert away_travel_penalty("LA", "SF", "13:00", neutral_site=True) is False


def test_late_kickoff_does_not_flag_even_cross_country_non_neutral():
    """A West Coast team flying east for a LATE game isn't the early-
    body-clock spot -- confirms neutral_site didn't accidentally change
    the existing kickoff-hour logic."""
    assert away_travel_penalty("NE", "LV", "20:00") is False
