# live current-season play-by-play loader
"""Loads the current season's play-by-play for live scoring and grading.

Why this exists instead of calling nfl.import_pbp_data([season]) directly:
nfl_data_py's default (include_participation=True) also downloads nflverse's
*separate* pbp_participation release for the season and merges it in. That
release lags the main play-by-play file -- for 2026 it did not exist at all
in Week 2 -- so the merge 404s and the whole load dies, and nfl_data_py's own
error handler for it references an undefined name (`except Error`) and raises
NameError instead of a readable message. Week 1 never noticed because it
scores off the prior-season fallback and never fetches current-season pbp.

When that happens this loads the main file without participation and adds the
participation-only columns back as all-NaN, so every downstream reader sees
the same schema it was built against. The only participation column the
pipeline reads is `was_pressure` (off/def pressure-rate stats), which is
narrative-only -- model/train.py excludes it from FEATURE_COLS -- and NaN just
means "not measured yet" there.

The historical training fetch (data/fetch_historical.py) deliberately does NOT
use this: silently degrading training data would be worse than failing loudly.
"""

import nfl_data_py as nfl
import numpy as np
import pandas as pd

# The columns nfl_data_py merges in from the pbp_participation release
# (checked against a real 2025 load: full-minus-base, and identical to the
# extra columns in data/cache/pbp.parquet).
PARTICIPATION_ONLY_COLS = (
    "defenders_in_box", "defense_coverage_type", "defense_man_zone_type",
    "defense_names", "defense_numbers", "defense_personnel", "defense_players",
    "defense_positions", "n_defense", "n_offense", "nflverse_game_id",
    "ngs_air_yards", "number_of_pass_rushers", "offense_formation",
    "offense_names", "offense_numbers", "offense_personnel", "offense_players",
    "offense_positions", "players_on_play", "possession_team", "route",
    "time_to_throw", "was_pressure",
)

_warned_seasons: set[int] = set()


def load_season_pbp(season: int) -> pd.DataFrame:
    """One season's play-by-play (downcast), tolerating a missing participation
    release -- see the module docstring. Any other failure (e.g. the main pbp
    file itself not being published yet) still raises."""
    try:
        return nfl.import_pbp_data([season], downcast=True)
    except Exception as first_error:
        pbp = nfl.import_pbp_data([season], downcast=True, include_participation=False)
        if season not in _warned_seasons:
            _warned_seasons.add(season)
            print(f"Warning: nflverse participation data for {season} isn't available "
                  f"({type(first_error).__name__}); loaded play-by-play without it -- "
                  f"pressure-rate stats will be blank until it's published.")
        missing = [c for c in PARTICIPATION_ONLY_COLS if c not in pbp.columns]
        if missing:
            pbp = pd.concat([pbp, pd.DataFrame(np.nan, index=pbp.index, columns=missing)], axis=1)
        return pbp
