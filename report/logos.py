# maps team abbreviation → ESPN logo URL
"""Maps each team abbreviation to a logo URL. Every team has a current logo
(ESPN's CDN). A handful of teams also have a throwback logo on file, sourced
and verified individually (Wikipedia for the Patriots, SportsLogos.net for
the rest, since Wikipedia doesn't host standalone historic logo files for
most franchises) -- used in the report whenever available, per request.

Note: the SportsLogos.net URLs are a fan reference site, not an official or
guaranteed-stable source -- if one of these links ever breaks, get_logo_url
falls back to the current ESPN logo automatically.
"""

import nfl_data_py as nfl

from config import CURRENT_SEASON

ESPN_LOGO_URL = "https://a.espncdn.com/i/teamlogos/nfl/500/{abbr}.png"

# Verified via direct HTTP checks (content-type: image/*), not guessed.
THROWBACK_LOGOS = {
    # New England Patriots -- "Pat Patriot", 1961-1992
    "NE": "https://upload.wikimedia.org/wikipedia/en/0/0b/New_England_Patriots_logo_old.svg",
    # Philadelphia Eagles -- kelly green era, 1987-1995
    "PHI": "https://content.sportslogos.net/logos/7/167/full/philadelphia_eagles_logo_primary_19875445.png",
    # Tampa Bay Buccaneers -- "Bucco Bruce" creamsicle era, 1976-1996
    "TB": "https://content.sportslogos.net/logos/7/176/full/tampa_bay_buccaneers_logo_primary_19768001.png",
    # Washington -- Redskins spear logo, 1972-2019
    "WAS": "https://content.sportslogos.net/logos/7/168/full/1063.gif",
    # Atlanta Falcons -- "Dirty Bird" era, 1990-2002
    "ATL": "https://content.sportslogos.net/logos/7/173/full/atlanta_falcons_logo_primary_19895520.png",

    # These came from the user directly (no public source URL), saved locally
    # at report/assets/logos/{ABBR}.png. The path below is relative to
    # report/output/, where build_report.py saves the HTML.
    # New York Giants -- 1950s "leaping football player" logo
    "NYG": "../assets/logos/NYG.png",
    # Kansas City Chiefs -- 1970s "running Chief" logo
    "KC": "../assets/logos/KC.png",
    # Miami Dolphins -- 1966-1996 leaping dolphin logo
    "MIA": "../assets/logos/MIA.png",
    # New York Jets -- 1978-1997 "flying Jets" wordmark logo
    "NYJ": "../assets/logos/NYJ.png",
    # Tennessee Titans -- Houston Oilers oil-derrick logo (same franchise,
    # pre-relocation/rename; replaces the 1999-2017 flaming-sword logo
    # previously used here)
    "TEN": "../assets/logos/TEN.png",
    # Pittsburgh Steelers -- 1940s-50s "Steely McBeam"-precursor steelworker logo
    "PIT": "../assets/logos/PIT.png",
    # Buffalo Bills -- standing bison logo, 1970-1973
    "BUF": "../assets/logos/BUF.png",
    # Detroit Lions -- leaping lion logo, 1961-1969
    "DET": "../assets/logos/DET.png",
    # Chicago Bears -- bear-on-football logo, 1940s-1960s
    "CHI": "../assets/logos/CHI.png",
    # San Francisco 49ers -- "gunslinger" prospector logo, 1950s-1990s
    "SF": "../assets/logos/SF.png",
    # Los Angeles Rams -- ram's-head logo (no helmet horns), used at various points
    "LA": "../assets/logos/LA.png",
    # Seattle Seahawks -- 1976-2001 seahawk-head logo
    "SEA": "../assets/logos/SEA.png",
    # Denver Broncos -- "D" logo with bucking horse, 1993-1996
    "DEN": "../assets/logos/DEN.png",
}


def current_logo_urls() -> dict:
    """abbr -> current ESPN logo URL, for every team active this season
    (excludes legacy abbreviations like OAK/SD/STL that nfl_data_py's team
    list still carries for historical lookups)."""
    sched = nfl.import_schedules([CURRENT_SEASON])
    active_abbrs = set(sched["home_team"]) | set(sched["away_team"])
    return {abbr: ESPN_LOGO_URL.format(abbr=abbr.lower()) for abbr in active_abbrs}


def get_logo_url(abbr: str, prefer_throwback: bool = True) -> str:
    if prefer_throwback and abbr in THROWBACK_LOGOS:
        return THROWBACK_LOGOS[abbr]
    return ESPN_LOGO_URL.format(abbr=abbr.lower())
