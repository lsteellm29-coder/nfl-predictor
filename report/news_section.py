# renders the recent-news feed section, informational only
"""Live NFL news feed display (Live News & Expanded Odds spec) -- purely
a UI section, same role as report/props.py's cards: takes already-fetched,
already-classified headlines (data/fetch_news.py's fetch_news()) and
renders them as a simple, dense list rather than the card-grid system
report/cards.py uses for props/picks, since a news ticker reads better
dense than as big cards.

Only ever shows what data/fetch_news.py already classified as
injury/suspension/lineup/trade-relevant (is_actionable) -- general
commentary/previews/recaps are filtered out here, not because they're
uninteresting, but because this section exists specifically as an early-
warning feed, not a general NFL news reader. This is purely a display
layer: nothing here feeds back into model/train.py or
model/player_stats.py, matching the spec's own "informational/
supplementary" requirement.
"""

import datetime
import html

SECTION_TEMPLATE = """<details class="card-section" open>
  <summary>Recent News</summary>
  <div class="news-feed">{items}</div>
</details>"""

ITEM_TEMPLATE = """<a class="news-item" href="{url}" target="_blank" rel="noopener">
  <span class="news-category news-category-{category}">{category_label}</span>
  {teams_html}
  <span class="news-headline">{headline}</span>
  <span class="news-published">{published}</span>
</a>"""

CATEGORY_LABELS = {
    "injury": "Injury", "suspension": "Suspension", "lineup": "Lineup", "trade": "Trade/Signing",
}

NEWS_STYLE = """
.news-feed { display: flex; flex-direction: column; gap: 2px; margin-top: 8px; }
.news-item { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; padding: 10px 12px; border-radius: 8px; text-decoration: none; color: inherit; }
.news-item:hover { background: var(--paper, #F4F5F8); }
.news-category { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.02em; padding: 3px 8px; border-radius: 999px; flex-shrink: 0; }
.news-category-injury, .news-category-suspension { background: var(--negative-soft, #FBEAEA); color: var(--negative, #B23A3A); }
.news-category-lineup { background: var(--warning-soft, #FFF6D9); color: var(--warning, #8A6D00); }
.news-category-trade { background: var(--paper, #F4F5F8); color: var(--muted, #666E7D); }
.news-team-tag { font-size: 12px; font-weight: 700; color: var(--muted, #666E7D); flex-shrink: 0; }
.news-headline { font-size: 14.5px; color: var(--ink, #171A21); flex: 1; min-width: 200px; }
.news-published { font-size: 12px; color: var(--muted, #666E7D); flex-shrink: 0; white-space: nowrap; }
.news-empty { font-size: 14px; color: var(--muted, #666E7D); padding: 12px; }
"""


def _relative_time(published: str | None) -> str:
    """A short "Xh ago"/"Xd ago" string from ESPN's ISO timestamp, or the
    raw date if parsing fails -- avoids claiming a timezone-relative
    "just now" that could read wrong depending on when the report is
    actually viewed vs. generated."""
    if not published:
        return ""
    try:
        pub = datetime.datetime.fromisoformat(published.replace("Z", "+00:00"))
        hours = (datetime.datetime.now(datetime.timezone.utc) - pub).total_seconds() / 3600
        if hours < 1:
            return f"{max(int(hours * 60), 1)}m ago"
        if hours < 24:
            return f"{int(hours)}h ago"
        return f"{int(hours / 24)}d ago"
    except (ValueError, TypeError):
        return published[:10]


def news_section_html(articles: list[dict] | None) -> str:
    if not articles:
        return SECTION_TEMPLATE.format(
            items='<div class="news-empty">News feed unavailable this run.</div>')

    relevant = [a for a in articles if a.get("is_actionable")]
    if not relevant:
        return SECTION_TEMPLATE.format(
            items='<div class="news-empty">Nothing injury/lineup/trade-relevant in the latest check -- '
                  "see each game's own injury status for current player availability.</div>")

    items = []
    for a in relevant[:20]:  # caps a busy news day from dwarfing the rest of the page
        category = a.get("category") or "trade"
        teams_html = "".join(
            f'<span class="news-team-tag">{html.escape(t)}</span>' for t in (a.get("teams") or [])[:2])
        items.append(ITEM_TEMPLATE.format(
            url=html.escape(a.get("url") or "#"),
            category=category, category_label=CATEGORY_LABELS.get(category, category.title()),
            teams_html=teams_html,
            headline=html.escape(a.get("headline") or ""),
            published=_relative_time(a.get("published")),
        ))
    return SECTION_TEMPLATE.format(items="".join(items))
