# pull game-day forecast (wind speed, temp, precip) for outdoor stadiums
"""Weather for outdoor games -- dome/retractable-roof teams are skipped
entirely (no weather impact worth modeling).

Historical training data needs nothing new here: nfl_data_py's cached
schedules already carry actual recorded temp/wind per game. Only the live
prediction path needs a real fetch, via Open-Meteo (free, no API key,
https://open-meteo.com) keyed by each stadium's coordinates.
"""

import datetime

import requests

# Stadium coordinates for every team with an outdoor or retractable roof.
# Dome teams are omitted on purpose -- weather never reaches the field there
# regardless of what's happening outside, so there's nothing to look up.
# Whether a *specific* game is played outdoors is decided by the schedule's
# own `roof` field, not this list -- retractable-roof stadiums (e.g. DAL,
# HOU, IND) show up here so a forecast is available if the roof is ever
# reported open, but most of their games will be `roof == "closed"`.
STADIUM_COORDS = {
    "ARI": (33.5276, -112.2626), "ATL": (33.7554, -84.4008),
    "BAL": (39.2780, -76.6227), "BUF": (42.7738, -78.7870),
    "CAR": (35.2258, -80.8528), "CHI": (41.8623, -87.6167),
    "CIN": (39.0954, -84.5160), "CLE": (41.5061, -81.6995),
    "DAL": (32.7473, -97.0945), "DEN": (39.7439, -105.0201),
    "GB": (44.5013, -88.0622), "HOU": (29.6847, -95.4107),
    "IND": (39.7601, -86.1639), "JAX": (30.3239, -81.6373),
    "KC": (39.0489, -94.4839), "MIA": (25.9580, -80.2389),
    "NE": (42.0909, -71.2643), "NYG": (40.8135, -74.0745),
    "NYJ": (40.8135, -74.0745), "PHI": (39.9008, -75.1675),
    "PIT": (40.4468, -80.0158), "SEA": (47.5952, -122.3316),
    "SF": (37.4032, -121.9698), "TB": (27.9759, -82.5033),
    "TEN": (36.1665, -86.7713), "WAS": (38.9076, -76.8645),
}

# Beyond this, passing/kicking efficiency measurably degrades -- the
# threshold matters more than the raw number (Section spec, Phase 4).
HIGH_WIND_MPH = 15.0

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_forecast(team: str, kickoff: datetime.datetime) -> dict | None:
    """(temp_f, wind_mph) forecast for `team`'s stadium at the closest
    available hour to kickoff. `kickoff` is expected in Eastern time, same
    as nflverse's `gametime` field -- the API defaults to returning
    timestamps in UTC regardless of stadium location, so timezone is
    requested explicitly as America/New_York to match without needing our
    own UTC conversion math. None if the team isn't in STADIUM_COORDS
    (dome team) or the forecast doesn't reach that far out yet."""
    if team not in STADIUM_COORDS:
        return None
    lat, lon = STADIUM_COORDS[team]

    resp = requests.get(FORECAST_URL, params={
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,wind_speed_10m,precipitation_probability",
        "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
        "forecast_days": 16, "timezone": "America/New_York",
    }, timeout=15)
    resp.raise_for_status()
    data = resp.json()["hourly"]

    times = [datetime.datetime.fromisoformat(t) for t in data["time"]]
    if not times or kickoff < times[0] or kickoff > times[-1]:
        return None  # outside the forecast window

    idx = min(range(len(times)), key=lambda i: abs((times[i] - kickoff).total_seconds()))
    return {
        "temp": data["temperature_2m"][idx],
        "wind": data["wind_speed_10m"][idx],
        "precip_prob": data["precipitation_probability"][idx],
    }
