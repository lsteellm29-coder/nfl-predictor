# data/fetch_week.py: a dead Odds API key must degrade to nflverse lines (and say
# so), never kill the run -- and never print the key.
import numpy as np
import pandas as pd
import pytest
import requests

import data.fetch_week as fw
from data.odds_http import OddsAPIError, raise_for_odds_status


def _schedule():
    rows = []
    for week, home, away, spread, total, ml_h, ml_a in [
        (2, "BUF", "DET", 5.5, 51.5, -238, 195),
        (2, "ATL", "CAR", -2.5, 43.5, 120, -140),
        (3, "KC", "BUF", 2.5, 50.0, -140, 120),
    ]:
        rows.append({
            "game_id": f"2026_{week:02d}_{away}_{home}", "season": 2026, "week": week, "game_type": "REG",
            "gameday": "2026-09-20", "gametime": "13:00", "weekday": "Sunday", "away_team": away,
            "home_team": home, "home_rest": 7, "away_rest": 7, "div_game": 0, "roof": "outdoors",
            "location": "Home", "away_qb_id": "x", "away_qb_name": "x", "home_qb_id": "y", "home_qb_name": "y",
            "away_coach": "c", "home_coach": "d", "spread_line": spread, "total_line": total,
            "home_moneyline": ml_h, "away_moneyline": ml_a,
            "home_score": np.nan, "away_score": np.nan,
        })
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def schedule(monkeypatch):
    monkeypatch.setattr(fw.nfl, "import_schedules", lambda seasons: _schedule())


def _dead_key(*_):
    resp = requests.Response()
    resp.status_code = 401
    resp._content = b'{"message":"API key is deactivated","error_code":"DEACTIVATED_KEY"}'
    resp.url = "https://api.the-odds-api.com/v4/x/odds/?apiKey=SECRET123"
    raise_for_odds_status(resp)


def test_error_message_has_status_and_code_but_never_the_url_or_key():
    with pytest.raises(OddsAPIError) as excinfo:
        _dead_key()
    msg = str(excinfo.value)
    assert "401" in msg and "DEACTIVATED_KEY" in msg
    assert "SECRET123" not in msg and "apiKey" not in msg and "http" not in msg
    assert isinstance(excinfo.value, requests.HTTPError)      # existing except-clauses still catch it
    assert excinfo.value.response.status_code == 401


def test_falls_back_to_nflverse_lines_when_the_odds_api_fails(monkeypatch, capsys):
    monkeypatch.setattr(fw, "fetch_odds", _dead_key)
    games = fw.fetch_week(2, 2026)

    assert set(games["lines_source"]) == {"nflverse"}
    assert len(games) == 2                                    # week 2 only -- not the whole season's lines
    buf = games[games["home_team"] == "BUF"].iloc[0]
    assert buf["spread_line"] == 5.5 and buf["total_line"] == 51.5
    assert buf["home_moneyline"] == -238 and buf["away_moneyline"] == 195
    # de-vigged: probabilities of the two sides sum to 1, so home (-238) is just under raw 70.4%
    assert 0.66 < buf["home_ml_prob"] < 0.70
    assert buf["event_id"] is None                            # -> props code skips per-game market lines
    out = capsys.readouterr().out
    assert "unavailable" in out and "DEACTIVATED_KEY" in out and "SECRET123" not in out


def test_falls_back_when_no_key_is_configured(monkeypatch):
    def no_key(season):
        raise RuntimeError("ODDS_API_KEY not set (expected in .env)")
    monkeypatch.setattr(fw, "fetch_odds", no_key)
    assert set(fw.fetch_week(2, 2026)["lines_source"]) == {"nflverse"}


def test_uses_the_odds_api_when_it_works_and_tags_the_source(monkeypatch):
    live = pd.DataFrame([{"home_team": "BUF", "away_team": "DET", "event_id": "abc", "n_books": 9,
                          "spread_line": 6.0, "total_line": 52.0, "home_moneyline": -250,
                          "away_moneyline": 205, "home_ml_prob": 0.7, "spread_n_books": 9, "total_n_books": 9}])
    monkeypatch.setattr(fw, "fetch_odds", lambda season: live)
    games = fw.fetch_week(2, 2026)
    assert set(games["lines_source"]) == {"odds_api"}
    assert games.set_index("home_team").at["BUF", "spread_line"] == 6.0     # the live number, not nflverse's 5.5
    assert np.isnan(games.set_index("home_team").at["ATL", "spread_line"])  # a game the feed lacks stays unlined


# ---- the key must not appear in NETWORK-error messages either ------------------------------------

def _network_down(*args, **kwargs):
    # what requests really raises: the message embeds the full URL, key included
    raise requests.ConnectionError(
        "HTTPSConnectionPool(host='api.the-odds-api.com', port=443): Max retries exceeded with url: "
        "/v4/sports/americanfootball_nfl/odds/?apiKey=SECRET123&regions=us%2Cus2 (Caused by "
        "NameResolutionError(\"Failed to resolve 'api.the-odds-api.com'\"))")


_URL = "https://api.the-odds-api.com/v4/x"
_PARAMS = {"apiKey": "SECRET123"}   # module level so the literal isn't in any traceback frame's source line


def test_odds_get_hides_the_key_in_connection_errors(monkeypatch):
    import traceback

    import data.odds_http as odds_http
    monkeypatch.setattr(odds_http.requests, "get", _network_down)
    with pytest.raises(OddsAPIError) as excinfo:
        odds_http.odds_get(_URL, _PARAMS)
    rendered = "".join(traceback.format_exception(excinfo.value))      # what a crash would print
    assert "SECRET123" not in rendered and "apiKey=" not in rendered
    assert "ConnectionError" in str(excinfo.value)                     # the failure kind is still visible
    assert isinstance(excinfo.value, requests.RequestException)        # still caught by existing handlers


def test_fetch_week_falls_back_without_leaking_the_key_when_the_network_is_down(monkeypatch, capsys):
    import data.odds_http as odds_http
    monkeypatch.setattr(fw, "ODDS_API_KEY", "SECRET123")
    monkeypatch.setattr(odds_http.requests, "get", _network_down)
    monkeypatch.setattr(fw, "_team_name_to_abbr", lambda season: {})

    games = fw.fetch_week(2, 2026)

    assert set(games["lines_source"]) == {"nflverse"}
    out = capsys.readouterr().out
    assert "SECRET123" not in out and "apiKey" not in out
