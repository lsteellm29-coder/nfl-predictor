# canonical NFL team-code normalization, shared across every data source
"""Week 1 Audit & Tuning Plan, Phase 1.1: nfl_data_py's cached data uses
whichever team abbreviation was actually in use in a given season, not a
single retroactively-normalized code across this project's whole
HISTORICAL_SEASONS window. Verified empirically against nfl_data_py's own
schedule data, 1999-2025:

  2016:      OAK, SD,  LA   (Raiders in Oakland, Chargers in San Diego,
                              Rams just relocated STL -> LA)
  2017-2019: OAK, LAC, LA
  2020-2025: LV,  LAC, LA

Three real relocations therefore carry two different code strings across
this codebase's cached window (STL->LA happened one season before
HISTORICAL_SEASONS starts, so it isn't live today, but is included here
for when that window is ever extended further back).

This silently broke three things before this module existed, all
confirmed by reading the actual grouping/lookup logic, not guessed:
model/elo.py's compute_elo_ratings() (a plain dict keyed by the raw code
string, so a relocation resets that franchise's entire accumulated Elo
history to the 1500 default, as if it were an expansion team --
model/train.py's training data was learning from a corrupted Elo signal
for years after each relocation); data/team_history.py's
team_last_n_meetings() (silently undercounts real head-to-head history
across a code change); and data/team_stats.py's ats_win_pct_last5 (the
one rolling stat deliberately grouped by team alone, not team+season, so
it resets to a cold start for a few games right after a relocation).

Separately, data/rosters.py already carries its own AZ->ARI alias (a
different kind of quirk -- nfl_data_py's CURRENT-season roster/snap-count
endpoints specifically, not a historical relocation) -- consolidated
into this same canonical map so there's one source of truth instead of
two that could drift apart.
"""

import pandas as pd

TEAM_CODE_ALIASES = {
    "OAK": "LV",    # Raiders, relocated to Las Vegas for the 2020 season
    "SD": "LAC",    # Chargers, relocated to Los Angeles for the 2017 season
    "STL": "LA",    # Rams, relocated to Los Angeles for the 2016 season
    "AZ": "ARI",    # Cardinals -- nfl_data_py's current-season roster/
                    # snap-count pull uses "AZ" while every other dataset
                    # (schedules, historical rosters, snap counts for
                    # already-completed seasons) uses "ARI"; see
                    # data/rosters.py's own docstring for how this was found.
}

# Every column name, across this codebase's dataframes, that ever holds a
# team abbreviation -- used as the default target set when the caller
# doesn't pass an explicit column list.
TEAM_CODE_COLUMNS = [
    "team", "home_team", "away_team", "posteam", "defteam", "opponent",
    "recovery_team", "penalty_team", "td_team", "return_team",
]


def normalize_team_codes(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Rewrites every known historic team-code alias to its canonical
    current code, in whichever of TEAM_CODE_COLUMNS actually exist on
    `df` (or the explicit `columns` list, if given -- useful for a
    dataframe with a team-code column under some other name). Returns a
    new dataframe; never mutates the input in place, so a caller that
    still holds a reference to the original frame doesn't get a surprise."""
    cols = columns if columns is not None else [c for c in TEAM_CODE_COLUMNS if c in df.columns]
    if not cols:
        return df
    df = df.copy()
    for col in cols:
        df[col] = df[col].replace(TEAM_CODE_ALIASES)
    return df
