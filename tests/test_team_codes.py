# tests for team-code normalization (Week 1 Audit & Tuning Plan Phase 1.1)
"""data/team_codes.py exists because nfl_data_py's cached data uses
whichever team abbreviation was actually in use in a given season (OAK
through 2019, LV from 2020 on, for the same Raiders franchise) rather
than a single retroactively-normalized code across this project's whole
HISTORICAL_SEASONS window. Confirmed via a real audit that this silently
broke model/elo.py's cross-season Elo continuity, data/team_history.py's
head-to-head lookups, and data/team_stats.py's ats_win_pct_last5 --
all three key off the raw team-code string with no idea two different
strings mean the same franchise.
"""

import pandas as pd

from data.team_codes import TEAM_CODE_ALIASES, normalize_team_codes


def test_normalizes_known_historic_relocations():
    df = pd.DataFrame({
        "home_team": ["OAK", "SD", "STL", "AZ"],
        "away_team": ["KC", "DEN", "SF", "DAL"],
    })
    result = normalize_team_codes(df)
    assert result["home_team"].tolist() == ["LV", "LAC", "LA", "ARI"]
    # away_team never had an alias in this fixture, must pass through unchanged
    assert result["away_team"].tolist() == ["KC", "DEN", "SF", "DAL"]


def test_does_not_touch_non_aliased_codes():
    df = pd.DataFrame({"team": ["KC", "SEA", "BUF"]})
    result = normalize_team_codes(df)
    assert result["team"].tolist() == ["KC", "SEA", "BUF"]


def test_only_normalizes_columns_that_actually_exist():
    """A dataframe missing every known team-code column (e.g. player-
    level data with no team column at all) must pass through unchanged,
    not raise a KeyError."""
    df = pd.DataFrame({"player_id": ["P1", "P2"], "yards": [10, 20]})
    result = normalize_team_codes(df)
    pd.testing.assert_frame_equal(result, df)


def test_normalizes_posteam_and_defteam_for_pbp_shaped_data():
    df = pd.DataFrame({"posteam": ["OAK"], "defteam": ["SD"], "play_type": ["pass"]})
    result = normalize_team_codes(df)
    assert result["posteam"].iloc[0] == "LV"
    assert result["defteam"].iloc[0] == "LAC"


def test_explicit_columns_argument_overrides_default_set():
    df = pd.DataFrame({"winner": ["OAK"], "loser": ["SD"]})
    result = normalize_team_codes(df, columns=["winner", "loser"])
    assert result["winner"].iloc[0] == "LV"
    assert result["loser"].iloc[0] == "LAC"


def test_does_not_mutate_the_input_dataframe():
    df = pd.DataFrame({"team": ["OAK"]})
    normalize_team_codes(df)
    assert df["team"].iloc[0] == "OAK"  # original untouched


def test_every_alias_target_is_itself_stable():
    """A canonical code must never appear as an alias KEY -- otherwise a
    second pass of normalization (or a dataframe that already had a mix
    of pre- and post-normalized rows) could double-translate a code that
    was already correct."""
    canonical_targets = set(TEAM_CODE_ALIASES.values())
    aliased_sources = set(TEAM_CODE_ALIASES.keys())
    assert canonical_targets.isdisjoint(aliased_sources)
