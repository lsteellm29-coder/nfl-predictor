# data/nfl_net.py: retry network failures in nfl_data_py's downloads, but never permanent ones.
import socket
import types
import urllib.error

import pytest

from data import nfl_net
from data.nfl_net import install_retries, is_transient, retrying


def _http(code):
    return urllib.error.HTTPError("https://x/y.parquet", code, "msg", {}, None)


def _name_error_from(inner):
    """What nfl_data_py really raises: a NameError from inside its own `except` block."""
    try:
        try:
            raise inner
        except Exception:
            raise NameError("name 'Error' is not defined")
    except NameError as e:
        return e


@pytest.mark.parametrize("error, expected", [
    (_http(504), True), (_http(503), True), (_http(429), True),
    (_http(404), False), (_http(403), False),
    (urllib.error.URLError(socket.gaierror(8, "nodename nor servname provided")), True),
    (ConnectionResetError("reset by peer"), True), (TimeoutError("slow"), True),
    (ValueError("bad data"), False), (KeyError("col"), False),
])
def test_is_transient(error, expected):
    assert is_transient(error) is expected


def test_is_transient_sees_through_nfl_data_pys_broken_error_handler():
    assert is_transient(_name_error_from(_http(504))) is True                     # a 504 wrapped in a NameError
    assert is_transient(_name_error_from(ConnectionResetError("reset"))) is True
    assert is_transient(_name_error_from(_http(404))) is False                    # e.g. participation not published
    assert is_transient(NameError("genuinely undefined")) is False


def test_retries_transient_errors_then_succeeds():
    calls, waits = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise _name_error_from(_http(504))
        return "ok"

    assert retrying(flaky, attempts=4, delay=2, sleep=waits.append)() == "ok"
    assert len(calls) == 3 and waits == [2, 4]


def test_does_not_retry_a_permanent_error():
    calls = []

    def missing():
        calls.append(1)
        raise _name_error_from(_http(404))

    with pytest.raises(NameError):
        retrying(missing, attempts=4, delay=1, sleep=lambda s: pytest.fail("must not sleep"))()
    assert len(calls) == 1                                                        # failed fast


def test_gives_up_after_the_last_attempt():
    calls = []

    def down():
        calls.append(1)
        raise ConnectionResetError("reset")

    with pytest.raises(ConnectionResetError):
        retrying(down, attempts=3, delay=0, sleep=lambda s: None)()
    assert len(calls) == 3


def test_install_wraps_import_functions_once_and_only_those():
    module = types.SimpleNamespace(import_schedules=lambda seasons: seasons, import_pbp_data=lambda s: s,
                                   clean_nfl_data=lambda df: df, VERSION="1.0")
    assert sorted(install_retries(module)) == ["import_pbp_data", "import_schedules"]
    assert install_retries(module) == []                                          # idempotent
    assert module.import_schedules([2026]) == [2026]                              # still behaves
    assert getattr(module.import_schedules, "_retrying", False)
    assert not getattr(module.clean_nfl_data, "_retrying", False)


def test_config_installs_retries_on_the_real_library():
    import config  # noqa: F401  (import side effect under test)
    import nfl_data_py as nfl
    assert getattr(nfl.import_pbp_data, "_retrying", False)
    assert getattr(nfl.import_schedules, "_retrying", False)
