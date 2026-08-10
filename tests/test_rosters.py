# regression test for the stale-roster-team bug (Combined Build Plan Part 1)
"""data/rosters.py's fetch_rosters() exists because several call sites
across this codebase pull a player_id x season roster spanning two
seasons (`[season, season - 1]`) and then build a lookup keyed off
whichever row a naive dict(zip(...)) or unsorted filter happened to keep
last -- which silently prefers the OLDER season whenever nfl_data_py
returns this season's rows first (it does). This was caught live:
nfl_data_py's current-season roster pull uses "AZ" for Arizona while
every other dataset this codebase reads still uses "ARI", so every
returning Cardinal's team was resolving to last season's "ARI" row by
accident of row order, not by any real current-season preference. This
test reproduces that exact shape with a synthetic two-season pull and
confirms the current season always wins, regardless of row order.
"""

import pandas as pd

import data.rosters as rosters_module


def _two_season_pull(row_order="current_first"):
    """One player who changed team/position between seasons (P1), one
    player only in the older season (P2, simulating someone not yet in
    this season's cut-down roster -- the reason a two-season pull
    happens at all), one player only in the current season (P3, a new
    signee). Deliberately uses team codes OUTSIDE TEAM_CODE_ALIASES for
    this test -- the season-preference bug and the team-code-alias bug
    are two different fixes in the same function; test_fetch_rosters_
    canonicalizes_known_team_code_aliases below covers the alias
    specifically, so this test isn't muddying the season-order
    assertion with a value fetch_rosters is now expected to rewrite."""
    current = pd.DataFrame([
        {"season": 2026, "player_id": "P1", "player_name": "Player One", "team": "KC", "position": "RB", "espn_id": "1"},
        {"season": 2026, "player_id": "P3", "player_name": "Player Three", "team": "SEA", "position": "WR", "espn_id": "3"},
    ])
    prior = pd.DataFrame([
        {"season": 2025, "player_id": "P1", "player_name": "Player One", "team": "TB", "position": "FB", "espn_id": "1"},
        {"season": 2025, "player_id": "P2", "player_name": "Player Two", "team": "NE", "position": "TE", "espn_id": "2"},
    ])
    if row_order == "current_first":
        return pd.concat([current, prior], ignore_index=True)
    return pd.concat([prior, current], ignore_index=True)


def test_fetch_rosters_prefers_current_season_regardless_of_row_order(monkeypatch):
    for row_order in ("current_first", "prior_first"):
        monkeypatch.setattr(
            rosters_module.nfl, "import_seasonal_rosters",
            lambda seasons, _order=row_order: _two_season_pull(_order))

        result = rosters_module.fetch_rosters([2026, 2025])
        by_id = result.set_index("player_id")

        # The core regression check: P1 (present in both seasons) must
        # resolve to the CURRENT season's team/position, never the
        # prior one -- this must hold no matter which order the two
        # seasons' rows happen to come back in.
        assert by_id.loc["P1", "team"] == "KC", f"row_order={row_order}"
        assert by_id.loc["P1", "position"] == "RB", f"row_order={row_order}"

        # A player only in last season (not yet in this season's pull)
        # still resolves via the fallback row -- the whole reason a
        # two-season pull happens.
        assert by_id.loc["P2", "team"] == "NE"

        # A player only in the current season resolves normally.
        assert by_id.loc["P3", "team"] == "SEA"

        # Exactly one row per player_id -- no duplicate/ambiguous rows
        # left over from the two-season concatenation.
        assert result["player_id"].is_unique


def test_fetch_rosters_drops_blank_player_id(monkeypatch):
    blank_id_row = pd.DataFrame([
        {"season": 2026, "player_id": "", "player_name": "No ID Guy", "team": "SEA", "position": "WR", "espn_id": None},
        {"season": 2026, "player_id": "P1", "player_name": "Player One", "team": "KC", "position": "RB", "espn_id": "1"},
    ])
    monkeypatch.setattr(rosters_module.nfl, "import_seasonal_rosters", lambda seasons: blank_id_row)

    result = rosters_module.fetch_rosters([2026])
    assert "" not in set(result["player_id"])
    assert len(result) == 1


def test_fetch_rosters_canonicalizes_known_team_code_aliases(monkeypatch):
    """The second, deeper bug found chasing the first: nfl_data_py's
    current-season roster pull uses "AZ" for Arizona while every other
    dataset this codebase joins against (schedules, snap counts, prior
    seasons) uses "ARI". Fixing only the season-preference bug actually
    made this WORSE -- a returning Cardinal correctly resolved to their
    current row, but that row's team ("AZ") then matched nothing when
    compared against schedule-sourced "ARI", silently dropping real
    current-roster players from anything that filters by team. This
    confirms fetch_rosters() rewrites a known alias before returning,
    so every caller can compare its `team` column against a schedule's
    home_team/away_team without a separate translation step."""
    raw = pd.DataFrame([
        {"season": 2026, "player_id": "P1", "player_name": "Cardinal Player", "team": "AZ", "position": "WR", "espn_id": "1"},
    ])
    monkeypatch.setattr(rosters_module.nfl, "import_seasonal_rosters", lambda seasons: raw)

    result = rosters_module.fetch_rosters([2026])
    assert result.set_index("player_id").loc["P1", "team"] == "ARI"
    assert "AZ" not in set(result["team"])
