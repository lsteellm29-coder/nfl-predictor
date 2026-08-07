# team-specific beat news + official depth charts via Firecrawl
"""MCP Integration spec, Section 1 (Firecrawl). Firecrawl turns web pages
into clean markdown -- used here for two things structured APIs (ESPN,
nfl_data_py, The Odds API) don't cover:

1. Team-specific news (data/fetch_news.py's ESPN pull is the league-wide
   national feed, not per-team beat coverage -- a roster move like
   "Eagles waive DE Ta'Quon Graham" shows up on the team's own site
   often before or instead of the national feed).
2. Official team depth charts (ourlads.com), as an independent second
   read on "who's actually rostered where" to cross-check against
   qa/validate_rosters.py's nfl_data_py-based roster -- same spirit as
   that module's existing balldontlie cross-check, a different source
   that can catch a case nfl_data_py's own preseason team-assignment
   data gets wrong (see model/player_stats.py's required_lineup
   docstring on that known gap).

Every source here was checked against robots.txt before being added:
ESPN's own robots.txt explicitly disallows the anthropic-ai user-agent
(among other AI crawlers), so ESPN is deliberately NOT a Firecrawl
target here -- data/fetch_news.py and data/fetch_injuries.py already get
ESPN data through its dedicated public JSON API instead, a different
(and already-established, pre-existing) integration this module doesn't
touch. NFL team sites (all on the same NFL Digital Media platform,
confirmed by sampling several) and ourlads.com both have permissive
robots.txt with no AI-crawler-specific blocks.
"""

import re
import time

import requests

from config import FIRECRAWL_API_KEY
from data.fetch_news import CATEGORY_PATTERNS

FIRECRAWL_SCRAPE_URL = "https://api.firecrawl.dev/v1/scrape"
# Free-tier limit observed live (11 req/min); Firecrawl doesn't send a
# Retry-After header on 429, only a "retry after Ns" note in the JSON
# error body -- same "parse what the API actually gives you, don't
# guess" approach data/fetch_balldontlie.py already uses for its own
# 429 handling.
MAX_RETRIES = 5
_RETRY_AFTER_PATTERN = re.compile(r"retry after (\d+)s")

# team abbreviation (nflverse convention) -> official site's news listing
# page. All 32 teams share the same NFL Digital Media platform and the
# same permissive robots.txt (verified by sampling PHI, SEA, NE, KC, BUF,
# DAL, SF, BAL directly), so this list is a config, not a guess -- add or
# remove teams here, nothing else needs to change.
TEAM_NEWS_URLS = {
    "ARI": "https://www.azcardinals.com/news/",
    "ATL": "https://www.atlantafalcons.com/news/",
    "BAL": "https://www.baltimoreravens.com/news/",
    "BUF": "https://www.buffalobills.com/news/",
    "CAR": "https://www.panthers.com/news/",
    "CHI": "https://www.chicagobears.com/news/",
    "CIN": "https://www.bengals.com/news/",
    "CLE": "https://www.clevelandbrowns.com/news/",
    "DAL": "https://www.dallascowboys.com/news/",
    "DEN": "https://www.denverbroncos.com/news/",
    "DET": "https://www.detroitlions.com/news/",
    "GB": "https://www.packers.com/news/",
    "HOU": "https://www.houstontexans.com/news/",
    "IND": "https://www.colts.com/news/",
    "JAX": "https://www.jaguars.com/news/",
    "KC": "https://www.chiefs.com/news/",
    "LA": "https://www.therams.com/news/",
    "LAC": "https://www.chargers.com/news/",
    "LV": "https://www.raiders.com/news/",
    "MIA": "https://www.miamidolphins.com/news/",
    "MIN": "https://www.vikings.com/news/",
    "NE": "https://www.patriots.com/news/",
    "NO": "https://www.neworleanssaints.com/news/",
    "NYG": "https://www.giants.com/news/",
    "NYJ": "https://www.newyorkjets.com/news/",
    "PHI": "https://www.philadelphiaeagles.com/news/",
    "PIT": "https://www.steelers.com/news/",
    "SEA": "https://www.seahawks.com/news/",
    "SF": "https://www.49ers.com/news/",
    "TB": "https://www.buccaneers.com/news/",
    "TEN": "https://www.tennesseetitans.com/news/",
    "WAS": "https://www.washingtoncommanders.com/news/",
}

DEPTH_CHART_URL = "https://www.ourlads.com/nfldepthcharts/depthchart/{team}"

# Ourlads uses the same team abbreviations as nflverse for every team
# except these -- confirmed by checking their team-picker links directly
# rather than assumed.
OURLADS_ABBR_FIX = {"LA": "LAR"}

_HEADLINE_PATTERN = re.compile(
    r"\*\*(.+?)\*\*[\s\\]*?([A-Za-z]{3} \d{1,2}, \d{4})\]\((https://[^\s)\"]+)")


def _firecrawl_scrape(url: str, formats: list[str]) -> dict:
    """Retries on 429 honoring Firecrawl's own "retry after Ns" message
    (falls back to a flat 15s wait if that message ever changes shape)
    -- without this, scraping all 32 teams in one run blew through the
    free-tier's 11-req/min limit after the first 8-11 calls and silently
    lost the rest (confirmed live before adding this retry loop)."""
    if not FIRECRAWL_API_KEY:
        raise RuntimeError("FIRECRAWL_API_KEY not set (expected in .env)")
    for attempt in range(MAX_RETRIES):
        resp = requests.post(
            FIRECRAWL_SCRAPE_URL,
            headers={"Authorization": f"Bearer {FIRECRAWL_API_KEY}", "Content-Type": "application/json"},
            json={"url": url, "formats": formats},
            timeout=30,
        )
        if resp.status_code == 429:
            m = _RETRY_AFTER_PATTERN.search(resp.json().get("error", ""))
            wait = int(m.group(1)) + 1 if m else 15
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()["data"]
    raise RuntimeError(f"Firecrawl: still rate-limited after {MAX_RETRIES} retries ({url})")


def _classify(headline: str) -> str | None:
    for category in ("injury", "suspension", "lineup", "trade", "coaching"):
        if CATEGORY_PATTERNS[category].search(headline):
            return category
    return None


def _to_iso_date(date_str: str) -> str:
    """"Aug 05, 2026" -> "2026-08-05T00:00:00Z" -- report/news_section.py's
    _relative_time() (and anything else consuming this module's output
    alongside data/fetch_news.py's ESPN articles) expects the same ISO
    shape ESPN's own API already returns; without this conversion,
    datetime.fromisoformat() fails silently and falls back to a raw
    string slice that truncates "Aug 05, 2026" into the meaningless
    "Aug 05, 20" (confirmed live before adding this)."""
    import datetime
    return datetime.datetime.strptime(date_str, "%b %d, %Y").strftime("%Y-%m-%dT00:00:00Z")


def scrape_team_news(team: str, url: str) -> list[dict]:
    """Every article on `team`'s news listing page, in the same shape
    data/fetch_news.py's fetch_news() returns (headline/description/
    published/url/teams/athlete_espn_ids/category/is_actionable) so
    run_week.py and build_artifact.py can merge both sources without
    caring which one a given headline came from. `athlete_espn_ids` is
    always empty here -- team news pages don't tag players the way
    ESPN's own news API does, and guessing at a player from headline
    text alone risks a wrong id more than it's worth; category-only
    tagging is still real signal on its own."""
    data = _firecrawl_scrape(url, ["markdown"])
    seen_urls = set()
    articles = []
    for headline, date, article_url in _HEADLINE_PATTERN.findall(data["markdown"]):
        if article_url in seen_urls:
            continue
        seen_urls.add(article_url)
        category = _classify(headline)
        try:
            published = _to_iso_date(date)
        except ValueError:
            published = None
        articles.append({
            "id": article_url, "headline": headline, "description": None,
            "published": published, "url": article_url, "teams": [team], "athlete_espn_ids": [],
            "category": category, "is_actionable": category in {"injury", "suspension", "lineup", "trade"},
        })
    return articles


def fetch_all_team_news(teams: list[str] | None = None) -> list[dict]:
    """Scrapes every team in `teams` (default: all 32) -- a single slow
    or broken team site shouldn't take down the whole refresh, so a
    per-team failure is skipped and logged, not raised."""
    articles = []
    for team, url in TEAM_NEWS_URLS.items():
        if teams is not None and team not in teams:
            continue
        try:
            articles.extend(scrape_team_news(team, url))
        except (requests.RequestException, KeyError) as e:
            print(f"fetch_firecrawl_sources: couldn't scrape {team} news ({e}), skipping.")
    return articles


_DEPTH_CHART_ROW = re.compile(r"^\|\s*([A-Z0-9]{1,4})\s*\|(.+)\|\s*$")
_DEPTH_CHART_PLAYER = re.compile(r"\[([A-Za-z .'\-]+),\s*([A-Za-z .'\-]+)\s+[^\]]*\]")


def scrape_team_depth_chart(team: str) -> dict:
    """{position: [player names, starter first]} from ourlads.com's own
    depth chart for `team` -- an independently-maintained second read on
    roster/starter assignment, not derived from nflverse/nfl_data_py at
    all, so it can catch a case that source gets wrong (see this
    module's docstring)."""
    ourlads_team = OURLADS_ABBR_FIX.get(team, team)
    data = _firecrawl_scrape(DEPTH_CHART_URL.format(team=ourlads_team), ["markdown"])
    chart: dict[str, list[str]] = {}
    for line in data["markdown"].splitlines():
        m = _DEPTH_CHART_ROW.match(line.strip())
        if not m:
            continue
        position = m.group(1)
        if position in ("Pos", "---"):
            continue
        players = [f"{first.strip()} {last.strip()}" for last, first in _DEPTH_CHART_PLAYER.findall(m.group(2))]
        if players:
            chart[position] = players
    return chart


def _normalize_name(name: str) -> str:
    """Same normalization model/player_stats.py's own _normalize_name
    applies to market-source names -- lowercased, periods stripped,
    Jr/Sr/II/III/IV suffix stripped. ourlads.com prints some players in
    ALL CAPS (no discernible pattern for which ones) and its own suffix
    conventions don't always match nfl_data_py's, so a raw string-equality
    check against roster_names produced 634 false-positive flags for a
    single team alone in this module's first real test -- correctly-
    rostered players like "Kyle Pitts Sr." (ourlads) vs. "Kyle Pitts"
    (nfl_data_py) and "JAKE MATTHEWS" vs. "Jake Matthews" flagged as
    "missing" when they weren't. Kept as a local copy rather than an
    import from model/player_stats.py to keep this module and the qa/
    package it feeds self-contained, same precedent
    data/fetch_balldontlie.py already set for this codebase's other
    independent-source cross-checks."""
    import re
    name = name.replace(".", "")
    name = re.sub(r"\s+(jr|sr|ii|iii|iv)$", "", name, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", name).strip().lower()


def cross_check_depth_chart(team: str, roster_names: set[str]) -> list[str]:
    """Flags any ourlads.com depth-chart player for `team` who doesn't
    appear at all in `roster_names` (nfl_data_py's own roster names for
    that team), matched after _normalize_name on both sides -- informational,
    same as qa/validate_rosters.py's balldontlie cross-check: a real
    discrepancy here means one of the two independent sources has this
    player on the wrong team, worth a human look, not something this
    check can resolve on its own. Normalizing both sides means this can
    still miss a genuine case where the two sources have wildly
    different spellings of the same person, but that tradeoff is the
    same one model/player_stats.py already accepted for market-name
    matching -- an unmatched name silently drops out rather than being
    guessed at, never a wrong flag presented as confident."""
    normalized_roster = {_normalize_name(n) for n in roster_names}
    chart = scrape_team_depth_chart(team)
    flagged = []
    for position, players in chart.items():
        for player in players:
            if _normalize_name(player) not in normalized_roster:
                flagged.append(f"{player} ({position}) listed on {team}'s ourlads.com depth chart, "
                                f"not found on {team}'s nfl_data_py roster")
    return flagged
