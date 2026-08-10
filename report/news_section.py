# renders the News tab's dedicated, filterable feed, informational only
"""Live NFL news feed display (Live News & Expanded Odds spec; SharpLine
Combined Build Plan Part 5). Purely a UI section, same role as
report/props.py's cards: takes already-fetched, already-classified
headlines (data/fetch_news.py's fetch_news()) and renders them as a
simple, dense list rather than the card-grid system report/cards.py uses
for props/picks, since a news feed reads better dense than as big cards.

Renders every article data/fetch_news.py managed to classify into one of
its five categories (injury/suspension/lineup/trade/coaching) -- this
used to be a small inline widget capped at 20 items and restricted to
just the four "actionable" categories (an early-warning ticker embedded
inline on the Predictions view); now that it's its own dedicated News
tab a viewer navigates to on purpose, that cap and restriction no longer
serve a purpose -- coaching-change headlines are exactly the kind of
thing report/theme.py's Combined Build Plan Part 3 (team change
tracking) makes newsworthy, and there's no more "dwarfing the rest of
the page" risk once this has its own tab. Genuinely uncategorized
articles (general commentary/previews/recaps `classify()` couldn't tag
at all) are still excluded -- this stays a football-news-relevant feed,
not a general reader.

This is purely a display layer: nothing here feeds back into
model/train.py or model/player_stats.py, matching the spec's own
"informational/supplementary" requirement.
"""

import datetime
import html

CATEGORY_LABELS = {
    "injury": "Injury", "suspension": "Suspension", "lineup": "Lineup",
    "trade": "Trade/Signing", "coaching": "Coaching",
}

NEWS_TAB_TEMPLATE = """<div class="filter-bar news-filter-bar">{filters}</div>
<div class="news-feed news-feed-tab">{items}</div>"""

ITEM_TEMPLATE = """<div class="news-item-wrap" data-category="{category}" data-teams="{data_teams}">
  <a class="news-item" href="{url}" target="_blank" rel="noopener">
    <span class="news-category news-category-{category}">{category_label}</span>
    {teams_html}
    <span class="news-headline">{headline}</span>
    <span class="news-published">{published}</span>
  </a>
  {cross_link}
</div>"""

NEWS_STYLE = """
.news-feed { display: flex; flex-direction: column; gap: 2px; margin-top: 8px; }
.news-item-wrap { display: flex; align-items: center; gap: 4px; }
.news-item { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; padding: 10px 12px; border-radius: 8px; text-decoration: none; color: inherit; flex: 1; min-width: 0; }
.news-item:hover { background: var(--paper, #F4F5F8); }
.news-category { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.02em; padding: 3px 8px; border-radius: 999px; flex-shrink: 0; }
.news-category-injury, .news-category-suspension { background: var(--negative-soft, #FBEAEA); color: var(--negative, #B23A3A); }
.news-category-lineup { background: var(--warning-soft, #FFF6D9); color: var(--warning, #8A6D00); }
.news-category-trade, .news-category-coaching { background: var(--paper, #F4F5F8); color: var(--muted, #666E7D); }
.news-team-tag { font-size: 12px; font-weight: 700; color: var(--muted, #666E7D); flex-shrink: 0; }
.news-headline { font-size: 14.5px; color: var(--ink, #171A21); flex: 1; min-width: 200px; }
.news-published { font-size: 12px; color: var(--muted, #666E7D); flex-shrink: 0; white-space: nowrap; }
.news-empty { font-size: 14px; color: var(--muted, #666E7D); padding: 12px; }
.news-cross-link { font-size: 12px; font-weight: 600; color: var(--accent, #A8710F); text-decoration: none; white-space: nowrap; padding: 6px 10px; flex-shrink: 0; }
.news-cross-link:hover { text-decoration: underline; }
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


def _team_to_anchor(predictions) -> dict:
    """team abbreviation -> that team's game anchor id (theme.py's
    "g-{away}-{home}" format), for this week's slate only -- a news
    item can only cross-link to a prediction card that actually exists
    on the page this run."""
    if predictions is None or predictions.empty:
        return {}
    from report.theme import game_anchor
    mapping = {}
    for _, g in predictions.iterrows():
        anchor = game_anchor(g)
        mapping[g["home_team"]] = anchor
        mapping[g["away_team"]] = anchor
    return mapping


def news_tab_html(articles: list[dict] | None, predictions=None) -> str:
    """`predictions` (model/predict.py's score_week() output, optional)
    powers the news-item -> prediction-card cross-link (Combined Build
    Plan Part 5): only added when a headline tags exactly one team --
    two-team headlines are already ambiguous about which of the two
    games/players it's really about, so this doesn't guess."""
    if not articles:
        return '<div class="news-empty">News feed unavailable this run.</div>'

    relevant = [a for a in articles if a.get("category") in CATEGORY_LABELS]
    if not relevant:
        return ('<div class="news-empty">Nothing categorized in the latest check -- '
                "see each game's own injury status for current player availability.</div>")

    team_anchor = _team_to_anchor(predictions)
    teams = sorted({t for a in relevant for t in (a.get("teams") or [])})
    categories = [c for c in CATEGORY_LABELS if any(a["category"] == c for a in relevant)]

    filters = []
    for c in categories:
        filters.append(f'<button type="button" class="filter-btn" data-filter-type="category" '
                        f'data-filter-value="{c}">{CATEGORY_LABELS[c]}</button>')
    for t in teams:
        filters.append(f'<button type="button" class="filter-btn" data-filter-type="team" '
                        f'data-filter-value="{t}">{t}</button>')

    items = []
    for a in relevant[:200]:  # a generous cap, not the old widget's 20 -- this is now the dedicated view
        category = a.get("category")
        item_teams = a.get("teams") or []
        teams_html = "".join(
            f'<span class="news-team-tag">{html.escape(t)}</span>' for t in item_teams[:2])
        cross_link = ""
        if len(item_teams) == 1 and item_teams[0] in team_anchor:
            cross_link = (f'<a class="news-cross-link" href="#{team_anchor[item_teams[0]]}">'
                          f'View prediction &rarr;</a>')
        items.append(ITEM_TEMPLATE.format(
            category=category, category_label=CATEGORY_LABELS.get(category, category.title()),
            data_teams=html.escape(" ".join(item_teams)),
            teams_html=teams_html,
            url=html.escape(a.get("url") or "#"),
            headline=html.escape(a.get("headline") or ""),
            published=_relative_time(a.get("published")),
            cross_link=cross_link,
        ))
    return NEWS_TAB_TEMPLATE.format(filters="".join(filters), items="".join(items))
