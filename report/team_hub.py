# per-team hub: season trend + recent TD-prop history for this week's teams
"""Round 2 UX, item #58: a closer look at a single team than the per-game
cards give -- logo/record, this week's matchup, a season-long rolling
EPA/play trend, and each of its top skill players' last-few-games TD-prop
trend (predicted vs. actually scored).

Scope decision (season not yet started when this was built -- 2026 week 1
is preseason, so team_stats.parquet/schedules.parquet have zero 2026 rows
yet): hubs are built only for the 32 -> N teams actually playing this
week, not all 32. This site has no real multi-page routing (everything is
one document, same as report/archive.py's per-week <details> pattern), and
generating a hub for 18 teams on bye with nothing to show would be scope
for its own sake. The "season trend" chart honestly labels which season's
weekly data it's showing (most recently completed season with real rows,
e.g. "2025 Season") rather than implying it's a live in-progress trend
that doesn't exist yet -- same "no fabricated confidence" rule as every
other chart in this codebase.

Player selection is scoped to whichever players already have a TD prop on
this week's slate (report/props.py's existing roster-validated lineup),
not the full historical roster -- reuses player identity (player_id) and
position data props already carries instead of a second roster lookup.
"""

import html

import pandas as pd

from data.team_change_tracker import team_change_callouts
from report.charts import player_td_trend_chart, team_epa_trend_chart

TEAM_HUB_SECTION = """<div class="section-head" id="team-hubs"><span class="accent">&#9679;</span> Team Hubs</div>
<div class="intro" style="margin-top:0;">A closer look at each team playing this week -- season trend and recent TD-prop history for their top skill players.</div>
{body}"""

HUB_BLOCK = """<details class="card-section team-hub">
  <summary>
    <img src="{logo}" alt="{team}" class="team-hub-logo">
    <span class="team-hub-name">{team_full}</span>
    <span class="team-hub-record">{record}</span>
  </summary>
  <div class="props-panel">
    <div class="team-hub-next">Next: {opponent_full} &middot; {kickoff} &middot; {home_away}{spread_bit}</div>
    {change_callouts_html}
    {trend_chart}
    {players_html}
  </div>
</details>"""

PLAYER_BLOCK = """<div class="team-hub-player">
  <div class="team-hub-player-name">{name} <span class="team-hub-player-pos">{position}</span></div>
  {chart}
</div>"""

NO_HISTORY = '<div class="team-hub-player-empty">No game history yet to chart.</div>'

TEAM_HUB_STYLE = """
.team-hub summary { display: flex; align-items: center; gap: 10px; }
.team-hub-logo { height: 28px; width: 28px; object-fit: contain; flex-shrink: 0; }
.team-hub-name { font-family: 'Oswald', sans-serif; font-weight: 700; font-size: 15px; }
.team-hub-record { font-family: 'Inter', sans-serif; font-weight: 400; font-size: 12px; color: var(--muted);
  text-transform: none; letter-spacing: normal; }
.team-hub-next { font-size: 12.5px; color: var(--muted); margin-bottom: 10px; }
.team-hub-players { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 14px; margin-top: 10px; }
.team-hub-player { background: var(--surface-raised); border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; }
.team-hub-player-name { font-size: 13px; font-weight: 600; }
.team-hub-player-pos { color: var(--muted); font-weight: 400; font-size: 11.5px; }
.team-hub-player-empty { font-size: 12px; color: var(--muted); margin-top: 6px; }
.team-hub-callouts { display: flex; flex-direction: column; gap: 4px; margin-bottom: 10px; }
.team-hub-callout { font-size: 12px; color: var(--warning); background: var(--warning-soft);
  border: 1px solid var(--warning); border-radius: 6px; padding: 6px 9px; line-height: 1.4; }
"""


def _esc(s) -> str:
    return html.escape(str(s))


def _team_record(schedules_df: pd.DataFrame, team: str) -> str:
    """Final W-L(-T) for the most recently completed regular season this
    team has full results for. Returns "--" (never a fabricated 0-0) if
    schedules_df has nothing playable for this team yet."""
    if schedules_df.empty:
        return "--"
    played = schedules_df[
        (schedules_df["game_type"] == "REG")
        & ((schedules_df["home_team"] == team) | (schedules_df["away_team"] == team))
        & schedules_df["home_score"].notna()
    ]
    if played.empty:
        return "--"
    season = played["season"].max()
    season_games = played[played["season"] == season]
    wins = losses = ties = 0
    for _, g in season_games.iterrows():
        is_home = g["home_team"] == team
        team_score = g["home_score"] if is_home else g["away_score"]
        opp_score = g["away_score"] if is_home else g["home_score"]
        if team_score > opp_score:
            wins += 1
        elif team_score < opp_score:
            losses += 1
        else:
            ties += 1
    record = f"{wins}-{losses}" + (f"-{ties}" if ties else "")
    return f"{record}, {int(season)}"


def _team_trend_weekly(team_stats_df: pd.DataFrame, team: str) -> list[dict]:
    df = team_stats_df[team_stats_df["team"] == team]
    if df.empty:
        return []
    season = df["season"].max()
    season_df = df[df["season"] == season].sort_values("week")
    return [
        {"season": int(season), "week": int(r["week"]),
         "off_epa": r["off_epa_per_play_avg"], "def_epa": r["def_epa_per_play_avg"]}
        for _, r in season_df.iterrows()
    ]


def player_td_games(td_backtest_df: pd.DataFrame, player_id: str, n: int = 5) -> list[dict]:
    if not player_id or pd.isna(player_id):
        return []
    df = td_backtest_df[td_backtest_df["player_id"] == player_id].sort_values(["season", "week"]).tail(n)
    return [
        {"week": int(r["week"]), "predicted_prob": r["predicted_prob"], "actual_td": bool(r["actual_td"])}
        for _, r in df.iterrows()
    ]


MAX_HUB_PLAYERS = 3


def _hub_players(props_df: pd.DataFrame, team: str) -> list[tuple]:
    """Top MAX_HUB_PLAYERS by this week's own projection, not every
    player with a prop (a team can have 4-5) -- caught during browser
    testing: 32 hubs x every prop-eligible player was enough embedded
    mini-charts to stall the page's renderer even before considering
    report/charts.py's own lazy-wiring fix. Highest-projection first
    is also just the more useful default -- the players most likely to
    actually be relevant to a "should I bet this" question."""
    if props_df is None or props_df.empty:
        return []
    team_props = props_df[props_df["team"] == team].dropna(subset=["player_id"])
    team_props = team_props.drop_duplicates("player_id").sort_values("projection", ascending=False)
    return [(r["player_id"], r["player"], r["position"]) for _, r in team_props.head(MAX_HUB_PLAYERS).iterrows()]


def team_hub_html(team: str, row: dict, team_stats_df: pd.DataFrame, schedules_df: pd.DataFrame,
                   td_backtest_df: pd.DataFrame, props_df: pd.DataFrame,
                   coach_changes: dict | None = None, unit_turnover: dict | None = None,
                   notable_moves: dict | None = None) -> str:
    """row: the same per-game dict report/build_report.py's/build_artifact.py's
    _row_data() already built for this team's game this week -- reused
    rather than recomputed, so kickoff/full-name/logo formatting stays
    identical to the rest of the page.

    coach_changes/unit_turnover/notable_moves (Combined Build Plan
    Part 3 step 3's "Display" requirement): data/team_change_tracker.py's
    three dicts, keyed by team -- all default to None/{} so a caller
    that hasn't fetched this data yet still gets a working hub, just
    without the callout block."""
    is_home = team == row["home_team"]
    opponent_full = row["home_full"] if not is_home else row["away_full"]
    team_full = row["home_full"] if is_home else row["away_full"]
    logo = row["home_logo"] if is_home else row["away_logo"]
    spread_bit = f" &middot; spread {row['spread_line']}" if row.get("spread_line") not in (None, "--") else ""

    callouts = team_change_callouts(team, coach_changes or {}, unit_turnover or {}, notable_moves or {})
    callout_items = "".join(f'<div class="team-hub-callout">{_esc(c)}</div>' for c in callouts)
    change_callouts_html = f'<div class="team-hub-callouts">{callout_items}</div>' if callouts else ""

    trend_weekly = _team_trend_weekly(team_stats_df, team)
    trend_chart = team_epa_trend_chart(trend_weekly)

    player_blocks = []
    for pid, name, position in _hub_players(props_df, team):
        games = player_td_games(td_backtest_df, pid)
        chart = player_td_trend_chart(games) if games else NO_HISTORY
        player_blocks.append(PLAYER_BLOCK.format(name=_esc(name), position=_esc(position), chart=chart))

    players_html = (f'<div class="team-hub-players">{"".join(player_blocks)}</div>'
                     if player_blocks else "")

    return HUB_BLOCK.format(
        logo=logo, team=_esc(team), team_full=_esc(team_full),
        record=_esc(_team_record(schedules_df, team)),
        opponent_full=_esc(opponent_full), kickoff=_esc(row["kickoff"]),
        home_away="Home" if is_home else "Away", spread_bit=spread_bit,
        change_callouts_html=change_callouts_html,
        trend_chart=trend_chart, players_html=players_html,
    )


def team_hubs_html(predictions: pd.DataFrame, props: pd.DataFrame, team_stats_df: pd.DataFrame,
                    schedules_df: pd.DataFrame, td_backtest_df: pd.DataFrame, row_data_fn,
                    coach_changes: dict | None = None, unit_turnover: dict | None = None,
                    notable_moves: dict | None = None) -> str:
    """row_data_fn: a single-arg callable (game row -> the same dict
    passed to GAME_BLOCK.format(**...)) supplied by the caller, so this
    module never has to import report.build_report and risk a circular
    import (build_report/build_artifact both import team_hub, not the
    other way around). coach_changes/unit_turnover/notable_moves: see
    team_hub_html()'s own docstring."""
    if predictions.empty:
        return ""
    teams = sorted(set(predictions["home_team"]) | set(predictions["away_team"]))
    blocks = []
    for team in teams:
        game_rows = predictions[(predictions["home_team"] == team) | (predictions["away_team"] == team)]
        if game_rows.empty:
            continue
        row = row_data_fn(game_rows.iloc[0])
        blocks.append(team_hub_html(
            team, row, team_stats_df, schedules_df, td_backtest_df, props,
            coach_changes, unit_turnover, notable_moves))
    if not blocks:
        return ""
    return TEAM_HUB_SECTION.format(body="\n".join(blocks))
