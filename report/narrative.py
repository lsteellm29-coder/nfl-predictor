# picks the single most convincing supporting storyline for the model's pick
"""The narrative-lookup layer between the raw matchup/streak/history data
(data/positional_matchups.py, data/player_trends.py, data/team_history.py)
and the prose in report/build_report.py. None of this touches the win-
probability model -- it runs *after* model/predict.py has already decided
a winner, and only ever looks for a storyline that backs that specific
pick up. If nothing clean does, build_report.py's existing top-3-factor
stat explanation carries the preview on its own, same as before this
module existed.
"""

from data.player_trends import QB_TREND_THRESHOLD, RB_TREND_THRESHOLD
from data.positional_matchups import POSITION_GROUPS

# A positional mismatch score below this (combined offense-produced +
# defense-allowed EPA, see data/positional_matchups.py) is real but not
# dramatic enough to lead a preview with. Calibrated empirically, not
# guessed: computed every team-vs-team mismatch score league-wide for a
# full season and set this at roughly the 85th percentile (~0.56) --
# 0.25 sounded plausible but was actually sitting at the *50th*
# percentile, meaning half of all possible matchups would have "qualified"
# as a mismatch worth leading with.
POSITIONAL_NOTABLE = 0.55

# win_margin / n_meetings -- 0.6 means at least 4-of-5 or 3-of-3. A 3-2
# record isn't a storyline, it's a coin flip with a story attached.
TEAM_H2H_CLEAN_MARGIN_RATE = 0.6
COACH_H2H_CLEAN_MARGIN_RATE = 0.6
# Coaching h2h is the weakest signal of the four -- more trivia than
# predictive edge -- so it only wins the "most extreme" comparison when
# nothing else comes close.
COACH_H2H_PRIORITY_WEIGHT = 0.7

GROUP_LABEL = {
    "WR": "wide receivers", "TE": "tight ends",
    "RB": "running backs out of the backfield", "RUSH": "ground game",
}


def _positional_candidates(winner: str, mismatches: list[dict]) -> list[tuple[float, dict]]:
    out = []
    for m in mismatches:
        # A mismatch supports the winner either because their own offense
        # has the edge (team == winner, score clearly positive) or because
        # their defense has the edge against the loser's offense (opponent
        # == winner, score clearly negative -- a bad matchup for the loser).
        supports_winner = (m["team"] == winner and m["score"] >= POSITIONAL_NOTABLE) or (
            m["opponent"] == winner and m["score"] <= -POSITIONAL_NOTABLE
        )
        if supports_winner:
            out.append((abs(m["score"]) / POSITIONAL_NOTABLE, m))
    return out


def _streak_candidates(winner: str, loser: str, streaks: list[dict | None], threshold: float) -> list[tuple[float, dict]]:
    out = []
    for s in streaks:
        if s is None:
            continue
        supports_winner = (s["team"] == winner and s["direction"] == "hot") or (
            s["team"] == loser and s["direction"] == "cold"
        )
        if supports_winner:
            out.append((abs(s["diff"]) / threshold, s))
    return out


def _h2h_candidate(winner: str, h2h: dict | None, a_key: str, b_key: str,
                    clean_rate: float, priority_weight: float = 1.0) -> tuple[float, dict] | None:
    """Shared logic for team_h2h (a_key/b_key = "team_a"/"team_b") and
    coach_h2h (a_key/b_key = "coach_a"/"coach_b") -- both dicts have the
    same a_wins/b_wins/ties/n shape (data/team_history.py)."""
    if h2h is None:
        return None
    if h2h[a_key] == winner:
        winner_wins, other_wins = h2h["a_wins"], h2h["b_wins"]
    elif h2h[b_key] == winner:
        winner_wins, other_wins = h2h["b_wins"], h2h["a_wins"]
    else:
        return None
    margin_rate = (winner_wins - other_wins) / h2h["n"]
    if margin_rate < clean_rate:
        return None
    return (priority_weight * margin_rate / clean_rate, h2h)


def select_lead_narrative(
    winner: str, loser: str,
    mismatches: list[dict], qb_streaks: list[dict | None], rb_streaks: list[dict | None],
    team_h2h: dict | None, coach_h2h: dict | None,
) -> dict | None:
    """The single most extreme candidate that supports `winner`, across all
    four narrative types, or None if nothing clears its type's bar. Each
    type's severity is normalized against its own "just barely notable"
    threshold, so a score of ~1.0 means "right at the bar" and higher
    means progressively more extreme -- comparable enough across
    differently-scaled metrics (EPA, games, meeting records) to rank them
    against each other."""
    candidates = (
        _positional_candidates(winner, mismatches)
        + _streak_candidates(winner, loser, qb_streaks, QB_TREND_THRESHOLD)
        + _streak_candidates(winner, loser, rb_streaks, RB_TREND_THRESHOLD)
    )
    team = _h2h_candidate(winner, team_h2h, "team_a", "team_b", TEAM_H2H_CLEAN_MARGIN_RATE)
    if team:
        candidates.append(team)
    coach = _h2h_candidate(winner, coach_h2h, "coach_a", "coach_b", COACH_H2H_CLEAN_MARGIN_RATE, COACH_H2H_PRIORITY_WEIGHT)
    if coach:
        candidates.append(coach)

    if not candidates:
        return None
    _, best = max(candidates, key=lambda c: c[0])
    return best


def _phrase_positional(m: dict, full_name) -> str:
    team, opp = full_name(m["team"]), full_name(m["opponent"])
    label = GROUP_LABEL[m["group"]]
    if m["group"] == "RUSH":
        return (
            f"The clearest edge in this one might be up front: {team}'s ground game has been averaging "
            f"{m['off_epa']:+.2f} EPA per rush this season, going up against a {opp} front seven that's "
            f"allowing {m['def_epa']:+.2f} EPA per carry."
        )
    return (
        f"The clearest edge in this one might be through the air: {team}'s {label} have been averaging "
        f"{m['off_epa']:+.2f} EPA per target, and {opp}'s defense has allowed {m['def_epa']:+.2f} EPA per "
        f"target to {label} this season."
    )


def _phrase_qb_streak(s: dict, full_name) -> str:
    team = full_name(s["team"])
    if s["direction"] == "hot":
        return (
            f"{s['player']} has been red-hot: {team}'s starting QB is averaging {s['recent_avg']:+.2f} "
            f"EPA/dropback over his last four games, well above his {s['season_avg']:+.2f} season mark."
        )
    return (
        f"{s['player']} has cooled off lately: {team}'s starting QB has managed just {s['recent_avg']:+.2f} "
        f"EPA/dropback over his last four games, down from his {s['season_avg']:+.2f} season average."
    )


def _phrase_rb_streak(s: dict, full_name) -> str:
    team = full_name(s["team"])
    if s["direction"] == "hot":
        return (
            f"{s['player']} is heating up: {team}'s lead back is averaging {s['recent_avg']:+.2f} EPA per "
            f"carry over his last four games, well above his {s['season_avg']:+.2f} season average."
        )
    return (
        f"{s['player']} has gone cold: {team}'s lead back is down to {s['recent_avg']:+.2f} EPA per carry "
        f"over his last four games, off his {s['season_avg']:+.2f} season average."
    )


def _phrase_team_h2h(h: dict, winner: str, full_name) -> str:
    winner_full = full_name(winner)
    if h["team_a"] == winner:
        other, winner_wins, other_wins = h["team_b"], h["a_wins"], h["b_wins"]
    else:
        other, winner_wins, other_wins = h["team_a"], h["b_wins"], h["a_wins"]
    return (
        f"Recent history leans this way too: {winner_full} has won {winner_wins} of the last {h['n']} "
        f"meetings with {full_name(other)}."
    )


def _phrase_coach_h2h(h: dict, winner_coach: str) -> str:
    if h["coach_a"] == winner_coach:
        other, winner_wins, other_wins = h["coach_b"], h["a_wins"], h["b_wins"]
    else:
        other, winner_wins, other_wins = h["coach_a"], h["b_wins"], h["a_wins"]
    return f"There's a coaching angle here too: {winner_coach} is {winner_wins}-{other_wins} all-time against {other}."


def phrase_lead_narrative(candidate: dict, winner: str, winner_coach: str, full_name) -> str:
    phrasers = {
        "positional": lambda c: _phrase_positional(c, full_name),
        "qb_streak": lambda c: _phrase_qb_streak(c, full_name),
        "rb_streak": lambda c: _phrase_rb_streak(c, full_name),
        "team_h2h": lambda c: _phrase_team_h2h(c, winner, full_name),
        "coach_h2h": lambda c: _phrase_coach_h2h(c, winner_coach),
    }
    return phrasers[candidate["type"]](candidate)
