# tests for the QB quality delta accuracy experiment (post-freeze follow-up to Phase 4.2)
import pandas as pd

from model.tune_qb_quality import MIN_DROPBACKS_FOR_QUALITY, _qb_quality, _qb_season_epa, build_qb_quality_dataset


def _pbp(rows):
    return pd.DataFrame(rows)


def test_qb_season_epa_averages_across_every_team_a_qb_played_for():
    """A midseason trade means the same QB appears under two different
    posteam values in one season -- the season-level average must still
    pool both, not silently drop one team's dropbacks."""
    pbp = _pbp([
        {"season": 2022, "posteam": "NYJ", "qb_dropback": 1, "passer_player_id": "QB1", "epa": 0.2},
        {"season": 2022, "posteam": "DEN", "qb_dropback": 1, "passer_player_id": "QB1", "epa": 0.4},
    ])
    result = _qb_season_epa(pbp)
    row = result[(result["season"] == 2022) & (result["passer_player_id"] == "QB1")].iloc[0]
    assert row["dropbacks"] == 2
    assert abs(row["epa"] - 0.3) < 1e-9


def test_qb_season_epa_excludes_non_dropback_plays():
    pbp = _pbp([
        {"season": 2022, "posteam": "NYJ", "qb_dropback": 1, "passer_player_id": "QB1", "epa": 0.2},
        {"season": 2022, "posteam": "NYJ", "qb_dropback": 0, "passer_player_id": "QB1", "epa": 99.0},
    ])
    result = _qb_season_epa(pbp)
    row = result[(result["season"] == 2022) & (result["passer_player_id"] == "QB1")].iloc[0]
    assert row["dropbacks"] == 1
    assert row["epa"] == 0.2


def test_qb_quality_none_below_the_dropback_floor():
    qb_season_epa = pd.DataFrame([{"season": 2022, "passer_player_id": "QB1", "epa": 0.3,
                                    "dropbacks": MIN_DROPBACKS_FOR_QUALITY - 1}])
    assert _qb_quality(qb_season_epa, 2022, "QB1") is None


def test_qb_quality_real_value_at_or_above_the_dropback_floor():
    qb_season_epa = pd.DataFrame([{"season": 2022, "passer_player_id": "QB1", "epa": 0.3,
                                    "dropbacks": MIN_DROPBACKS_FOR_QUALITY}])
    assert _qb_quality(qb_season_epa, 2022, "QB1") == 0.3


def test_qb_quality_none_for_a_qb_with_no_record_that_season():
    qb_season_epa = pd.DataFrame([{"season": 2022, "passer_player_id": "QB1", "epa": 0.3, "dropbacks": 300}])
    assert _qb_quality(qb_season_epa, 2022, "QB2") is None


def test_qb_quality_none_for_a_missing_qb_id():
    qb_season_epa = pd.DataFrame(columns=["season", "passer_player_id", "epa", "dropbacks"])
    assert _qb_quality(qb_season_epa, 2022, float("nan")) is None


def test_build_qb_quality_dataset_skips_games_missing_either_sides_prior_record():
    """A rookie or barely-played backup starting Week 1 has no qualifying
    PRIOR-season record -- that game must be excluded, not guessed at."""
    schedules = pd.DataFrame([
        {"season": 2022, "week": 1, "game_type": "REG", "home_team": "SEA", "away_team": "NE",
         "home_score": 24, "away_score": 17, "home_qb_id": "QB_H", "away_qb_id": "QB_A"},
    ])
    pbp = _pbp([
        {"season": 2021, "posteam": "SEA", "qb_dropback": 1, "passer_player_id": "QB_H", "epa": 0.1},
    ] * MIN_DROPBACKS_FOR_QUALITY)  # SEA's QB qualifies; NE's QB (QB_A) has no 2021 record at all
    result = build_qb_quality_dataset(schedules, pbp)
    assert result.empty


def test_build_qb_quality_dataset_computes_diff_as_home_minus_away():
    schedules = pd.DataFrame([
        {"season": 2022, "week": 1, "game_type": "REG", "home_team": "SEA", "away_team": "NE",
         "home_score": 24, "away_score": 17, "home_qb_id": "QB_H", "away_qb_id": "QB_A"},
    ])
    pbp = pd.concat([
        _pbp([{"season": 2021, "posteam": "SEA", "qb_dropback": 1, "passer_player_id": "QB_H", "epa": 0.3}]
             * MIN_DROPBACKS_FOR_QUALITY),
        _pbp([{"season": 2021, "posteam": "NE", "qb_dropback": 1, "passer_player_id": "QB_A", "epa": 0.1}]
             * MIN_DROPBACKS_FOR_QUALITY),
    ], ignore_index=True)
    result = build_qb_quality_dataset(schedules, pbp)
    assert len(result) == 1
    row = result.iloc[0]
    assert abs(row["qb_quality_diff"] - 0.2) < 1e-9
    assert row["margin"] == 7
