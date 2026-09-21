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


# ---- blowout flag must exist for games that haven't been played yet ---------------------------

def _live_season():
    import pandas as pd
    rows = [
        {"season": 2026, "game_type": "REG", "week": 1, "home_team": "NE", "away_team": "SEA",
         "home_score": 10.0, "away_score": 40.0},                                  # NE loses by 30
        {"season": 2026, "game_type": "REG", "week": 2, "home_team": "PIT", "away_team": "NE",
         "home_score": None, "away_score": None},                                  # not played yet
        {"season": 2026, "game_type": "REG", "week": 2, "home_team": "SEA", "away_team": "ARI",
         "home_score": None, "away_score": None},
    ]
    return pd.DataFrame(rows)


def test_blowout_flag_is_set_for_an_upcoming_game_after_a_blowout_loss():
    from data.situational import blowout_loss_flags
    flags = blowout_loss_flags(_live_season())
    wk2 = dict(zip(flags[flags["week"] == 2]["team"], flags[flags["week"] == 2]["blowout_loss_last_game"]))
    # this used to be {} (no row for any unplayed game) -> the live lookup defaulted every team to 0
    assert wk2 == {"NE": 1, "PIT": 0, "SEA": 0, "ARI": 0}


def test_blowout_flag_for_an_unplayed_game_uses_the_previous_game_not_the_unplayed_one():
    from data.situational import blowout_loss_flags
    flags = blowout_loss_flags(_live_season())
    ne = flags[flags["team"] == "NE"].sort_values("week")
    assert list(ne["blowout_loss_last_game"]) == [0, 1]       # week 1: no prior game; week 2: prior was a 30-pt loss
