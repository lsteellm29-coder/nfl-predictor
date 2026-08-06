# formats predictions + logos + lines into output
"""Formats a week's predictions into the weekly report: matchup (with team
logos), kickoff date/time, predicted winner, win %, Vegas line, edge, and a
plain-language explanation of what's actually driving each prediction --
grouped by day like a calendar, printed to the console and saved as an HTML
file (Section 1 / Section 5 of the spec).
"""

import datetime
import hashlib
import os
import textwrap

import nfl_data_py as nfl
import pandas as pd

from config import CURRENT_SEASON
from report.logos import get_logo_url

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# (higher_is_better, subject phrase, value format) per feature. Used to
# ground the "why" paragraph in real numbers instead of vague phrases.
# higher_is_better is checked against the *actual* fval/oval before using any
# comparative language ("well ahead of", "tighter than") -- collinear
# features can make the model credit a team on a stat where its raw number
# isn't actually better (see def_epa_per_play_avg_diff's coefficient sign
# flip from training), so asserting superiority unconditionally would
# sometimes state things backwards. When the raw numbers don't back up the
# model's attribution, this falls back to neutral "compared to" phrasing
# instead of a wrong claim.
FEATURE_INFO = {
    "points_for_avg_diff": (True, "{team} is averaging {v} points per game this season", "{:.1f}"),
    "points_against_avg_diff": (False, "{team}'s defense is allowing {v} points per game", "{:.1f}"),
    "off_epa_per_play_avg_diff": (True, "{team}'s offense is grading out at {v} EPA per play", "{:+.2f}"),
    "def_epa_per_play_avg_diff": (False, "{team}'s defense is allowing {v} EPA per play", "{:+.2f}"),
    "off_ypp_avg_diff": (True, "{team} is picking up {v} yards per play on offense", "{:.1f}"),
    "def_ypp_avg_diff": (False, "{team} is allowing {v} yards per play on defense", "{:.1f}"),
    "off_third_down_pct_avg_diff": (True, "{team} is converting {v} of third downs on offense", "{:.0%}"),
    "def_third_down_pct_avg_diff": (False, "{team}'s defense is allowing third-down conversions at a {v} rate", "{:.0%}"),
    "turnover_diff_avg_diff": (True, "{team} is at {v} in turnover margin per game", "{:+.2f}"),
    "red_zone_td_pct_avg_diff": (True, "{team} is scoring touchdowns on {v} of red-zone trips", "{:.0%}"),
    "ats_win_pct_season_diff": (True, "{team} has covered the betting spread in {v} of its games this season", "{:.0%}"),
    "ats_win_pct_last5_diff": (True, "{team} has covered in {v} of its last five games", "{:.0%}"),
    "injury_impact_diff": (False, "{team}'s injury report carries an impact score of {v}", "{:.1f}"),
}
COMPARATORS = ["well ahead of", "clearly better than", "a step above"]

# Deterministic phrasing pools so consecutive games don't read like
# copy-paste of each other -- picked per game from a stable hash, not
# randomly, so the same game always reads the same way on a re-run.
OPENERS = [
    "{winner} is the pick here, projected to win about {pct} of the time.",
    "This one leans toward {winner}, who the model has winning roughly {pct} of the time.",
    "The numbers favor {winner} in this matchup, with a win probability around {pct}.",
    "{winner} comes out on top of the model's projection, favored in about {pct} of simulated outcomes.",
]
TRANSITIONS = [
    "Still, it's not one-sided: {other} has its own case here -- ",
    "That said, {other} isn't without an argument: ",
    "{other} does have a real case, though: ",
    "It's worth noting {other} counters with something too: ",
]
TD_INTROS = [
    "On the touchdown-scorer side: {away_scorer} leads the way for {away} (about a {away_prob:.0%} shot at finding the end zone based on recent scoring rate), while {home_scorer} projects as {home}'s likeliest scorer (roughly {home_prob:.0%}).",
    "If you're watching anytime-TD markets, {home_scorer} has been {home}'s go-to scorer (~{home_prob:.0%} rate), and {away_scorer} has carried that role for {away} (~{away_prob:.0%}).",
    "Up front on scoring: {away_scorer} has been the one finding paydirt most often for {away} (~{away_prob:.0%}), with {home_scorer} doing the same for {home} (~{home_prob:.0%}).",
]
EDGE_BIG = [
    "For context: sportsbooks currently have {vegas_side} favored by about {vegas_val:.1f} points, but based on the team stats above, this model thinks the game is closer to {model_side} by {model_val:.1f}. That gap is what's sometimes called an 'edge' -- a sign the model sees this matchup differently than the sportsbooks do, in {favored_poss} favor.",
    "Here's the interesting part: the market has this at {vegas_side} by {vegas_val:.1f}, while this model's own number comes out to {model_side} by {model_val:.1f}. That difference suggests the sportsbooks may be pricing this one a bit off from what the underlying numbers say, in {favored_poss} favor.",
    "Worth flagging: Vegas has {vegas_side} favored by {vegas_val:.1f}, but the model's math lands on {model_side} by {model_val:.1f} instead -- a real enough gap to call it an edge, favoring {favored}.",
]
EDGE_SMALL = [
    "For context: this model's own read on the game ({model_side} by {model_val:.1f}) is close to what sportsbooks have it at ({vegas_side} by {vegas_val:.1f}), so there isn't much disagreement here.",
    "The model and the market are mostly aligned on this one -- {model_side} by {model_val:.1f} here versus {vegas_side} by {vegas_val:.1f} from sportsbooks -- so there's not much of an edge to point to.",
    "Not much daylight between the model and the betting line on this game: {model_side} by {model_val:.1f} versus the market's {vegas_side} by {vegas_val:.1f}.",
]

_TEAM_NAMES_CACHE = None


def _team_names() -> dict:
    global _TEAM_NAMES_CACHE
    if _TEAM_NAMES_CACHE is None:
        sched = nfl.import_schedules([CURRENT_SEASON])
        abbrs = set(sched["home_team"]) | set(sched["away_team"])
        team_desc = nfl.import_team_desc()
        current = team_desc[team_desc["team_abbr"].isin(abbrs)]
        _TEAM_NAMES_CACHE = dict(zip(current["team_abbr"], current["team_name"]))
    return _TEAM_NAMES_CACHE


def _full_name(abbr: str) -> str:
    return _team_names().get(abbr, abbr)


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>NFL Week {week} Predictions -- {season}</title>
<style>
  body {{ font-family: 'Segoe UI', -apple-system, Helvetica, Arial, sans-serif; background: #ffffff; color: #1c1c1e; margin: 0; padding: 32px; font-size: 17px; }}
  h1 {{ font-size: 30px; margin-bottom: 24px; font-weight: 700; }}
  h2 {{ font-size: 18px; text-transform: uppercase; letter-spacing: 0.04em; color: #555; margin: 40px 0 16px; border-bottom: 2px solid #e5e5e5; padding-bottom: 10px; font-weight: 700; }}
  .game {{ background: #fafafa; border: 1px solid #e2e2e2; border-radius: 14px; padding: 26px 30px; margin-bottom: 22px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }}
  .top-row {{ display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; }}
  .matchup {{ display: flex; align-items: center; gap: 16px; font-size: 24px; font-weight: 700; }}
  .matchup img {{ height: 52px; width: 52px; object-fit: contain; }}
  .coaches {{ font-size: 15px; color: #777; margin-top: 4px; }}
  .kickoff {{ font-size: 15px; color: #777; }}
  .stats {{ display: flex; gap: 26px; font-size: 16px; color: #333; margin-top: 18px; flex-wrap: wrap; }}
  .stats b {{ color: #111; }}
  .winner {{ font-weight: 700; color: #1e7d34; }}
  .edge-pos {{ color: #1e7d34; }}
  .edge-neg {{ color: #c62828; }}
  .edge-flat {{ color: #777; }}
  .why {{ font-size: 16.5px; color: #333; margin-top: 18px; line-height: 1.65; }}
  .caveat {{ font-size: 14px; color: #888; margin: -12px 0 28px; max-width: 720px; line-height: 1.5; }}
</style>
</head>
<body>
<h1>NFL Week {week} Predictions -- {season}</h1>
<div class="caveat">This model backtested at 61.4% accuracy on the 2025 season (vs. 53.6% for always picking the home team) -- a real edge, but far from certain. TD-scorer odds start from each player's own recent scoring rate, then adjust for the opposing defense's TDs-allowed rate and this game's Vegas-implied scoring environment. Treat all of this as informed estimates, not guarantees.</div>
{days}
</body>
</html>
"""

DAY_TEMPLATE = """<h2>{day_header}</h2>
{games}"""

GAME_TEMPLATE = """<div class="game">
  <div class="top-row">
    <div>
      <div class="matchup">
        <img src="{away_logo}" alt="{away_team}"> {away_full} @ {home_full}
        <img src="{home_logo}" alt="{home_team}">
      </div>
      <div class="coaches">{coaches}</div>
    </div>
    <div class="kickoff">{kickoff}</div>
  </div>
  <div class="stats">
    <div>Winner: <span class="winner">{winner}</span> ({win_pct})</div>
    <div>Vegas: <b>{spread_line}</b> / total {total_line}</div>
    <div>Moneyline: <b>{moneyline}</b></div>
    <div>Model: <b>{implied_spread}</b></div>
    <div>Edge: <b class="{edge_class}">{edge}</b></div>
  </div>
  <div class="stats">
    <div>Likely TD scorer ({away_team}): <b>{away_td_scorer}</b></div>
    <div>Likely TD scorer ({home_team}): <b>{home_td_scorer}</b></div>
  </div>
  <div class="why">{why}</div>
</div>"""


def _fmt_kickoff(gameday, gametime, weekday) -> str:
    if pd.isna(gameday):
        return "TBD"
    time_str = "TBD"
    if pd.notna(gametime):
        try:
            time_str = datetime.datetime.strptime(gametime, "%H:%M").strftime("%-I:%M %p ET")
        except ValueError:
            time_str = gametime
    return f"{weekday}, {gameday} -- {time_str}"


def _day_header(gameday: str, weekday: str) -> str:
    try:
        dt = datetime.datetime.strptime(gameday, "%Y-%m-%d")
        return f"{weekday}, {dt.strftime('%B %-d, %Y')}"
    except (ValueError, TypeError):
        return str(gameday)


def _possessive(name: str) -> str:
    return f"{name}'" if name.endswith("s") else f"{name}'s"


def _join(items: list) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _coach_qb_line(game: pd.Series) -> str:
    parts = []
    home_coach, away_coach = game.get("home_coach"), game.get("away_coach")
    home_qb, away_qb = game.get("home_qb_name"), game.get("away_qb_name")
    if pd.notna(away_coach):
        bit = f"{_full_name(game['away_team'])}: coached by {away_coach}"
        if pd.notna(away_qb):
            bit += f", starting {away_qb} at QB"
        parts.append(bit)
    if pd.notna(home_coach):
        bit = f"{_full_name(game['home_team'])}: coached by {home_coach}"
        if pd.notna(home_qb):
            bit += f", starting {home_qb} at QB"
        parts.append(bit)
    return "  |  ".join(parts)


def _game_seed(game: pd.Series) -> int:
    key = f"{game.get('season')}-{game.get('week')}-{game['home_team']}-{game['away_team']}"
    return int(hashlib.md5(key.encode()).hexdigest(), 16)


def _pick(pool: list, seed: int, salt: int) -> str:
    return pool[(seed + salt) % len(pool)]


def _cite_factor(feat: str, game: pd.Series, favored_abbr: str, other_abbr: str, seed: int = 0) -> str | None:
    """favored_abbr/other_abbr are team abbreviations (e.g. 'SEA'), not full
    names -- comparing against game['home_team']/['away_team'] (also
    abbreviations) has to use the same kind of value or every comparison
    silently fails and stats end up attributed to the wrong team."""
    home, away = game["home_team"], game["away_team"]
    favored_stats = game.get("home_stats") if favored_abbr == home else game.get("away_stats")
    other_stats = game.get("away_stats") if favored_abbr == home else game.get("home_stats")
    favored, other = _full_name(favored_abbr), _full_name(other_abbr)

    if feat == "rest_diff":
        home_rest, away_rest = game.get("home_rest"), game.get("away_rest")
        if pd.isna(home_rest) or pd.isna(away_rest):
            return None
        f_rest, o_rest = (home_rest, away_rest) if favored_abbr == home else (away_rest, home_rest)
        return f"{favored} is working with {f_rest:.0f} days' rest heading in, versus {o_rest:.0f} for {other}"

    if feat == "home_field_context_diff":
        h_val = (game.get("home_stats") or {}).get("home_point_diff_avg")
        a_val = (game.get("away_stats") or {}).get("away_point_diff_avg")
        if h_val is None or a_val is None or pd.isna(h_val) or pd.isna(a_val):
            return None
        home_full, away_full = _full_name(home), _full_name(away)
        return (f"{home_full} has outscored opponents by {h_val:+.1f} points per game at home "
                f"this season, while {away_full} is at {a_val:+.1f} per game on the road")

    info = FEATURE_INFO.get(feat)
    base = feat.replace("_diff", "")
    if not info or not favored_stats or not other_stats:
        return None
    higher_is_better, subject_template, value_fmt = info
    fval, oval = favored_stats.get(base), other_stats.get(base)
    if fval is None or oval is None or pd.isna(fval) or pd.isna(oval):
        return None

    favored_clause = subject_template.format(team=favored, v=value_fmt.format(fval))
    other_val_str = value_fmt.format(oval)
    favored_is_actually_better = (fval > oval) if higher_is_better else (fval < oval)
    if favored_is_actually_better:
        comparator = _pick(COMPARATORS, seed, sum(map(ord, feat)))
        return f"{favored_clause}, {comparator} {other}'s {other_val_str}"
    return f"{favored_clause}, compared to {other}'s {other_val_str}"


def _matchup_note(scorer: dict, scorer_team: str, opp_team: str) -> str | None:
    """Names the specific matchup reason when the opposing defense or the
    game's expected scoring environment moved a TD-scorer estimate
    meaningfully off the player's own raw rate."""
    def_factor, total_factor = scorer["def_factor"], scorer["total_factor"]
    opp_full = _full_name(opp_team)
    if def_factor > 1.1:
        return f"{opp_full}'s defense has been giving up scores at an above-average rate, which helps here"
    if def_factor < 0.9:
        return f"{opp_full}'s defense has been stingy about letting anyone score, which works against this pick"
    if total_factor > 1.1:
        return f"this game's Vegas total points to a higher-scoring environment than usual, which helps here"
    if total_factor < 0.9:
        return "this game's Vegas total points to a lower-scoring environment than usual, which works against this pick"
    return None


def _td_sentence(game: pd.Series, seed: int) -> str | None:
    home, away = game["home_team"], game["away_team"]
    home_td, away_td = game.get("home_td_scorer"), game.get("away_td_scorer")
    if not home_td or not away_td:
        return None
    template = _pick(TD_INTROS, seed, 2)
    sentence = template.format(
        home=_full_name(home), away=_full_name(away),
        home_scorer=home_td["player"], away_scorer=away_td["player"],
        home_prob=home_td["prob"], away_prob=away_td["prob"],
    )

    for scorer, scorer_team, opp_team in [(home_td, home, away), (away_td, away, home)]:
        note = _matchup_note(scorer, scorer_team, opp_team)
        if note:
            sentence += f" Worth noting: {note}."
            break  # one matchup callout is plenty; more starts to ramble

    return sentence


def _explain(game: pd.Series, winner: str) -> str:
    top_factors = game.get("top_factors")
    if not top_factors:
        return ("There isn't enough recent history on one of these teams yet "
                "(likely because their season just started) to explain a prediction here.")

    seed = _game_seed(game)
    home, away = game["home_team"], game["away_team"]
    home_full, away_full = _full_name(home), _full_name(away)
    winner_full = home_full if winner == home else away_full
    other = away if winner == home else home
    other_full = away_full if winner == home else home_full

    winner_cites, other_cites = [], []
    for feat, contrib in top_factors:
        side = home if contrib > 0 else away
        other_side = away if side == home else home
        cite = _cite_factor(feat, game, side, other_side, seed)
        if cite is None:
            continue
        (winner_cites if side == winner else other_cites).append(cite)

    sentences = [_pick(OPENERS, seed, 0).format(winner=winner_full, pct=f"{max(game['home_win_prob'], 1 - game['home_win_prob']) * 100:.0f}%")]

    if winner_cites:
        sentences.append(f"The biggest reasons why: {_join(winner_cites)}.")

    if other_cites:
        sentences.append(_pick(TRANSITIONS, seed, 1).format(other=other_full) + f"{_join(other_cites)}.")

    td_sentence = _td_sentence(game, seed)
    if td_sentence:
        sentences.append(td_sentence)

    edge = game.get("edge")
    vegas = game.get("spread_line")
    model_spread = game.get("implied_spread")
    if edge is not None and pd.notna(edge) and pd.notna(vegas):
        vegas_side = home_full if vegas > 0 else away_full
        model_side = home_full if model_spread > 0 else away_full
        favored = home_full if edge > 0 else away_full
        if abs(edge) > 1:
            sentences.append(_pick(EDGE_BIG, seed, 3).format(
                vegas_side=vegas_side, vegas_val=abs(vegas), model_side=model_side,
                model_val=abs(model_spread), favored=favored, favored_poss=_possessive(favored)))
        else:
            sentences.append(_pick(EDGE_SMALL, seed, 3).format(
                vegas_side=vegas_side, vegas_val=abs(vegas), model_side=model_side, model_val=abs(model_spread)))

    return " ".join(sentences)


def _fmt_spread(value) -> str:
    if pd.isna(value):
        return "--"
    return f"home {value:+.1f}"


def _fmt_moneyline(game: pd.Series) -> str:
    home_ml, away_ml = game.get("home_moneyline"), game.get("away_moneyline")
    if pd.isna(home_ml) or pd.isna(away_ml):
        return "--"
    home_str = f"{home_ml:+.0f}" if home_ml else "--"
    away_str = f"{away_ml:+.0f}" if away_ml else "--"
    return f"home {home_str} / away {away_str}"


def _fmt_td_scorer(pred: dict | None) -> str:
    if not pred:
        return "no recent TD data available"
    combined = pred["def_factor"] * pred["total_factor"]
    if combined > 1.08:
        adj_note = " -- bumped up for a favorable matchup"
    elif combined < 0.92:
        adj_note = " -- knocked down for a tough matchup"
    else:
        adj_note = ""
    return (f"{pred['player']} -- {pred['prob'] * 100:.0f}% anytime-TD estimate{adj_note} "
            f"({pred['td_count']} TDs in {pred['games']} games, {pred['source']})")


def _row_data(game: pd.Series) -> dict:
    has_pred = pd.notna(game.get("home_win_prob"))

    if has_pred:
        home_prob = game["home_win_prob"]
        if home_prob >= 0.5:
            winner, win_pct = game["home_team"], home_prob
        else:
            winner, win_pct = game["away_team"], 1 - home_prob
        win_pct_str = f"{win_pct * 100:.0f}%"
        implied_spread_str = f"{game['implied_spread']:+.1f}"
        edge = game["edge"]
        if edge is None or pd.isna(edge):
            edge_str, edge_class = "--", "edge-flat"
        else:
            edge_str = f"{edge:+.1f}"
            edge_class = "edge-pos" if edge > 1 else ("edge-neg" if edge < -1 else "edge-flat")
        why = _explain(game, winner)
    else:
        winner, win_pct_str, implied_spread_str = "--", "--", "--"
        edge_str, edge_class = "--", "edge-flat"
        why = "Not enough data on one of these teams yet to make a prediction."

    return {
        "away_team": game["away_team"],
        "home_team": game["home_team"],
        "away_full": _full_name(game["away_team"]),
        "home_full": _full_name(game["home_team"]),
        "away_logo": get_logo_url(game["away_team"]),
        "home_logo": get_logo_url(game["home_team"]),
        "coaches": _coach_qb_line(game),
        "kickoff": _fmt_kickoff(game.get("gameday"), game.get("gametime"), game.get("weekday")),
        "winner": winner,
        "win_pct": win_pct_str,
        "spread_line": _fmt_spread(game.get("spread_line")),
        "total_line": "--" if pd.isna(game.get("total_line")) else f"{game['total_line']:.1f}",
        "moneyline": _fmt_moneyline(game),
        "implied_spread": implied_spread_str,
        "edge": edge_str,
        "edge_class": edge_class,
        "away_td_scorer": _fmt_td_scorer(game.get("away_td_scorer")),
        "home_td_scorer": _fmt_td_scorer(game.get("home_td_scorer")),
        "why": why,
    }


def _by_day(predictions: pd.DataFrame):
    """Yields (gameday, weekday, games_df) in chronological order."""
    ordered_days = sorted(predictions["gameday"].dropna().unique())
    for day in ordered_days:
        day_games = predictions[predictions["gameday"] == day]
        weekday = day_games.iloc[0]["weekday"]
        yield day, weekday, day_games


def print_report(predictions: pd.DataFrame, week: int, season: int) -> None:
    print(f"\nNFL Week {week} Predictions -- {season}")
    print(textwrap.fill(
        "Model backtested at 61.4% accuracy on the 2025 season (vs. 53.6% for always "
        "picking the home team). TD-scorer odds are adjusted for the opposing defense's "
        "TDs-allowed rate and this game's Vegas-implied scoring environment. Treat as "
        "informed estimates, not guarantees.",
        width=78))
    print("=" * 78)
    for day, weekday, day_games in _by_day(predictions):
        print(f"\n{_day_header(day, weekday)}")
        print("-" * 78)
        for _, game in day_games.iterrows():
            d = _row_data(game)
            matchup = f"{d['away_team']} @ {d['home_team']}"
            print(f"{matchup:<12} {d['kickoff'].split('-- ')[-1]:<10} "
                  f"winner: {d['winner']:<4} ({d['win_pct']:>4})  "
                  f"vegas: {d['spread_line']:<10} ml: {d['moneyline']:<24} "
                  f"model: {d['implied_spread']:<6} edge: {d['edge']}")
            print(f"    Likely TD scorer ({d['away_team']}): {d['away_td_scorer']}")
            print(f"    Likely TD scorer ({d['home_team']}): {d['home_td_scorer']}")
            print(textwrap.fill(d["why"], width=78, initial_indent="    ", subsequent_indent="    "))


def build_html_report(predictions: pd.DataFrame, week: int, season: int) -> str:
    days_html = []
    for day, weekday, day_games in _by_day(predictions):
        games_html = "\n".join(GAME_TEMPLATE.format(**_row_data(g)) for _, g in day_games.iterrows())
        days_html.append(DAY_TEMPLATE.format(day_header=_day_header(day, weekday), games=games_html))
    return HTML_TEMPLATE.format(week=week, season=season, days="\n".join(days_html))


def save_report(predictions: pd.DataFrame, week: int, season: int = CURRENT_SEASON) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"week_{week}_{season}.html")
    with open(path, "w") as f:
        f.write(build_html_report(predictions, week, season))
    return path


def build_report(predictions: pd.DataFrame, week: int, season: int = CURRENT_SEASON) -> str:
    print_report(predictions, week, season)
    path = save_report(predictions, week, season)
    print(f"\nSaved report -> {path}")
    return path


def main():
    import argparse

    from model.predict import score_week

    parser = argparse.ArgumentParser()
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--season", type=int, default=CURRENT_SEASON)
    args = parser.parse_args()

    predictions = score_week(args.week, args.season)
    build_report(predictions, args.week, args.season)


if __name__ == "__main__":
    main()
