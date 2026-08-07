# live NFL news feed, tagged by team/player and classified for injury/lineup relevance
"""Live NFL News Feed spec. Pulls ESPN's news endpoint (same free/
unofficial API family already used for injuries -- see
data/fetch_injuries.py), tags each headline by team/player using the
feed's own category metadata, and classifies relevance with a lightweight
keyword matcher rather than a per-headline Claude API call: this pipeline
pulls dozens of headlines per refresh and runs unattended, so a real LLM
call per headline would add real latency and cost for a task a keyword
list handles well enough, and headline text is short and formulaic
("X ruled out", "Y elevated from practice squad") in exactly the ways
that make keyword matching viable.

This stays informational/supplementary by design (per the spec): nothing
here ever touches model/train.py's FEATURE_COLS or
model/player_stats.py's projections. It exists to (a) show users real
context via the news feed section in the report/artifact, and (b) flag
headlines worth a human/automated cross-check against the *structured*
injury and roster data this codebase already trusts for real -- an
early-warning layer ahead of the weekly injury report and inactive-list
pulls, never a replacement for either.
"""

import re

import requests

from data.fetch_injuries import ESPN_ABBR_FIX

ESPN_NEWS_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/news"

# Checked in this priority order since a headline can plausibly match more
# than one category (e.g. "released" also implying a lineup change) --
# earlier categories are the ones this pipeline most needs an early
# warning on, so they win a headline that matches several.
CATEGORY_KEYWORDS = {
    "injury": [
        r"\binjur\w*", r"\bconcussion", r"\bquestionable\b", r"\bdoubtful\b",
        r"\bruled out\b", r"\bout for\b", r"\binjured reserve\b", r"\bplaced on ir\b",
        r"\bday.to.day\b", r"\btorn \w+", r"\bsprain\w*", r"\bfracture\w*",
        r"\bsurgery\b", r"\bmiss(es|ed)? (the )?season\b", r"\bseason.ending\b",
    ],
    # Distinct from "injury" per the spec's own explicit list ("injuries,
    # trades, suspensions, coaching changes") -- functionally similar
    # (both mean a player's unavailable), but a suspension is a
    # discipline/eligibility issue, not a health one, so it's worth
    # keeping visually and semantically separate rather than folded in.
    "suspension": [
        r"\bsuspend\w*", r"\bban(ned)?\b", r"\bineligible\b",
    ],
    "lineup": [
        r"\belevated\b", r"\bpractice squad\b", r"\bactivated\b",
        r"\bnamed starter\b", r"\bwill start\b", r"\bbenched\b",
        r"\binactive\b", r"\bdepth chart\b", r"\bstarting job\b",
    ],
    "trade": [
        r"\btrade\w*", r"\bacquir\w*", r"\bwaive\w*", r"\breleas\w*",
        r"\bsign\w*", r"\bfree agen\w*", r"\bclaim\w*(ed|s)?\b",
    ],
    "coaching": [
        r"\bhead coach\b", r"\bfired\b", r"\bhired\b", r"\bcoordinator\b",
        r"\bcoaching staff\b", r"\binterim coach\b",
    ],
}
CATEGORY_PATTERNS = {cat: re.compile("|".join(kws), re.IGNORECASE) for cat, kws in CATEGORY_KEYWORDS.items()}

# Only these actually bear on a specific player's own availability/usage
# status -- a coaching change is real context but doesn't change what
# apply_injury_usage() should discount for anyone, so it isn't part of
# the roster/injury cross-check trigger, just shown in the feed.
ACTIONABLE_CATEGORIES = {"injury", "suspension", "lineup", "trade"}


def fetch_raw_news(limit: int = 50) -> list[dict]:
    resp = requests.get(ESPN_NEWS_URL, params={"limit": limit}, timeout=20)
    resp.raise_for_status()
    return resp.json().get("articles", [])


def _extract_tags(article: dict) -> tuple[list[str], list[str]]:
    """(team abbreviations, athlete ESPN ids) tagged on this article, from
    its own categories -- the only place ESPN's news feed links a
    headline back to specific teams/players, same idea as
    data/fetch_injuries.py's injury-report parsing."""
    teams, athlete_ids = [], []
    for cat in article.get("categories", []):
        if cat.get("type") == "team":
            abbr = (cat.get("team") or {}).get("abbreviation")
            if abbr:
                teams.append(ESPN_ABBR_FIX.get(abbr, abbr))
        elif cat.get("type") == "athlete":
            athlete_id = (cat.get("athlete") or {}).get("id")
            if athlete_id is not None:
                athlete_ids.append(str(athlete_id))
    return teams, athlete_ids


def classify(headline: str, description: str | None) -> str | None:
    """Best-matching category (injury/lineup/trade/coaching), or None if
    this headline doesn't look relevant to any of them -- general
    commentary, previews, recaps, etc. all correctly classify as None."""
    text = f"{headline} {description or ''}"
    for category in ("injury", "suspension", "lineup", "trade", "coaching"):
        if CATEGORY_PATTERNS[category].search(text):
            return category
    return None


def parse_articles(articles: list[dict]) -> list[dict]:
    rows = []
    for a in articles:
        teams, athlete_ids = _extract_tags(a)
        category = classify(a.get("headline", ""), a.get("description"))
        rows.append({
            "id": a.get("id"),
            "headline": a.get("headline"),
            "description": a.get("description"),
            "published": a.get("published"),
            "url": ((a.get("links") or {}).get("web") or {}).get("href"),
            "teams": teams,
            "athlete_espn_ids": athlete_ids,
            "category": category,
            "is_actionable": category in ACTIONABLE_CATEGORIES,
        })
    return rows


def fetch_news(limit: int = 50) -> list[dict]:
    """Every recent NFL headline, tagged and classified -- see this
    module's docstring: informational only, never a direct model input."""
    return parse_articles(fetch_raw_news(limit))


def cross_check_against_injuries(articles: list[dict], current_injury_status: dict, espn_to_gsis: dict) -> list[dict]:
    """Actionable headlines (injury/lineup/trade) tagging a player this
    codebase already has a roster entry for, flagged for review --
    `current_injury_status` is data/fetch_injuries.py's
    fetch_current_player_injury_status() output (GSIS id -> status dict);
    `espn_to_gsis` is that same module's espn_id_to_gsis_id() crosswalk.
    Each flag notes whether the tagged player is ALREADY reflected in the
    structured injury report or not -- a relevant headline about someone
    with no matching entry there yet is exactly the "catch it before the
    official report updates" case this module exists for. This never
    writes back into current_injury_status itself; a human or a
    downstream automated step decides what, if anything, to do with a
    flag."""
    flags = []
    for article in articles:
        if not article["is_actionable"]:
            continue
        for espn_id in article["athlete_espn_ids"]:
            gsis_id = espn_to_gsis.get(espn_id)
            if not gsis_id:
                continue
            flags.append({
                "headline": article["headline"], "category": article["category"],
                "player_espn_id": espn_id, "player_gsis_id": gsis_id,
                "published": article["published"], "url": article["url"],
                "already_on_injury_report": gsis_id in current_injury_status,
            })
    return flags


def main():
    from config import CURRENT_SEASON
    from data.fetch_injuries import espn_id_to_gsis_id, fetch_current_player_injury_status

    articles = fetch_news()
    print(f"Fetched {len(articles)} headlines.")
    actionable = [a for a in articles if a["is_actionable"]]
    print(f"{len(actionable)} classified as injury/lineup/trade-relevant:")
    for a in actionable[:15]:
        teams = ",".join(a["teams"]) or "?"
        print(f"  [{a['category']:8s}] ({teams}) {a['headline']}")

    injury_status = fetch_current_player_injury_status([CURRENT_SEASON, CURRENT_SEASON - 1])
    espn_to_gsis = espn_id_to_gsis_id([CURRENT_SEASON, CURRENT_SEASON - 1])
    flags = cross_check_against_injuries(articles, injury_status, espn_to_gsis)
    print(f"\n{len(flags)} flag(s) for review (relevant headline tagging a rostered player):")
    for f in flags:
        tracked = "already on injury report" if f["already_on_injury_report"] else "NOT yet on injury report"
        print(f"  [{f['category']:8s}] {f['headline']} -- {tracked}")


if __name__ == "__main__":
    main()
