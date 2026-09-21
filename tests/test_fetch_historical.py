# data/fetch_historical.py: a dropped connection must not take the weekly run down.
import pandas as pd
import pytest

import data.fetch_historical as fh


def _no_sleep(_seconds):
    pass


def test_retries_a_flaky_fetch_until_it_succeeds():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionResetError("Connection reset by peer")
        return "data"

    assert fh._with_retries(flaky, "pbp", attempts=4, delay=1, sleep=_no_sleep) == "data"
    assert len(calls) == 3


def test_gives_up_and_raises_after_the_last_attempt():
    calls = []

    def always_fails():
        calls.append(1)
        raise NameError("name 'Error' is not defined")     # what nfl_data_py's broken handler raises

    with pytest.raises(NameError):
        fh._with_retries(always_fails, "pbp", attempts=3, delay=1, sleep=_no_sleep)
    assert len(calls) == 3


def test_backoff_delay_grows():
    waits = []

    def always_fails():
        raise OSError("down")

    with pytest.raises(OSError):
        fh._with_retries(always_fails, "pbp", attempts=3, delay=5, sleep=waits.append)
    assert waits == [5, 10]


def test_refresh_keeps_the_existing_cache_when_the_download_fails(tmp_path, capsys):
    path = tmp_path / "pbp.parquet"
    pd.DataFrame({"x": [1, 2]}).to_parquet(path)
    before = path.read_bytes()

    def down():
        raise ConnectionResetError("reset")

    assert fh._refresh("play-by-play", down, str(path), attempts=2, delay=0, sleep=_no_sleep) is None
    assert path.read_bytes() == before                               # untouched, never partially written
    assert "keeping the existing cache" in capsys.readouterr().out


def test_refresh_raises_when_there_is_no_cache_to_fall_back_on(tmp_path):
    def down():
        raise ConnectionResetError("reset")

    with pytest.raises(ConnectionResetError):
        fh._refresh("play-by-play", down, str(tmp_path / "missing.parquet"), attempts=2, delay=0, sleep=_no_sleep)
    assert not (tmp_path / "missing.parquet").exists()


def test_refresh_writes_the_new_data_on_success(tmp_path):
    path = tmp_path / "schedules.parquet"
    df = fh._refresh("schedules", lambda: pd.DataFrame({"x": [1, 2, 3]}), str(path))
    assert len(df) == 3 and len(pd.read_parquet(path)) == 3
