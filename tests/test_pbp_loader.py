# data/pbp_loader.py: current-season pbp must survive a missing participation release.
import numpy as np
import pandas as pd
import pytest

import data.pbp_loader as pbp_loader
from data.pbp_loader import PARTICIPATION_ONLY_COLS, load_season_pbp


def _base_pbp():
    return pd.DataFrame({"week": [1, 1, 2], "epa": [0.1, -0.2, 0.3]})


def test_passes_through_when_participation_is_available(monkeypatch):
    frame = _base_pbp().assign(was_pressure=[0.0, 1.0, 0.0])
    monkeypatch.setattr(pbp_loader.nfl, "import_pbp_data", lambda seasons, downcast=True: frame)
    assert load_season_pbp(2026) is frame


def test_falls_back_without_participation_and_restores_the_schema(monkeypatch, capsys):
    calls = []

    def fake_import(seasons, downcast=True, include_participation=True):
        calls.append(include_participation)
        if include_participation:
            # what nfl_data_py actually does when pbp_participation_<year> 404s
            raise NameError("name 'Error' is not defined")
        return _base_pbp()

    monkeypatch.setattr(pbp_loader.nfl, "import_pbp_data", fake_import)
    monkeypatch.setattr(pbp_loader, "_warned_seasons", set())

    pbp = load_season_pbp(2026)

    assert calls == [True, False]
    assert list(pbp["epa"]) == [0.1, -0.2, 0.3]                       # real data untouched
    for col in PARTICIPATION_ONLY_COLS:
        assert col in pbp.columns and pbp[col].isna().all()          # schema restored as NaN
    assert len(pbp) == 3
    assert "participation data for 2026 isn't available" in capsys.readouterr().out


def test_still_raises_when_the_main_file_is_unavailable_too(monkeypatch):
    def fake_import(seasons, downcast=True, include_participation=True):
        raise OSError("main pbp file not published")

    monkeypatch.setattr(pbp_loader.nfl, "import_pbp_data", fake_import)
    with pytest.raises(OSError, match="main pbp file"):
        load_season_pbp(2026)


def test_participation_columns_match_what_the_cached_history_carries():
    """Guards the hard-coded list against drifting from the real data: every
    participation column must exist in the cached historical pbp, and was_pressure
    (the one downstream code reads) must be in the list."""
    import os
    import pyarrow.parquet as pq

    from data.live_stats import PBP_PATH

    assert "was_pressure" in PARTICIPATION_ONLY_COLS
    if not os.path.exists(PBP_PATH):
        pytest.skip("historical pbp cache not present")
    cached = set(pq.ParquetFile(PBP_PATH).schema_arrow.names)
    assert set(PARTICIPATION_ONLY_COLS) <= cached
