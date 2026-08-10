# auto-generated "how the model did" recap for whichever games/props just got graded
"""Master Honing Plan round 2, item #9: a short narrative recap generated
right after run_week.py's existing grading step (_grade_pending/
_grade_pending_props) finishes -- the biggest correct calls, the biggest
misses, and how the week went against the market overall, in the same
templated-but-varied prose voice report/build_report.py's _explain()
already uses (deterministic phrasing pools, not an LLM call -- this
codebase doesn't call out to any generative model anywhere in the
pipeline, and a recap is exactly the kind of "sounds authoritative but
could quietly misstate a real result" task that's worth keeping
deterministic and grounded in the actual graded numbers).

Only ever built from rows that were GENUINELY graded in the run that just
happened -- never from the full season history (that's the Track Record
page's job) and never from a still-pending row. A week with nothing
gradeable yet (mid-week, no games finished) produces no recap at all,
same "no fabricated content" contract as the rest of this project: no
recap is more honest than a recap about nothing.
"""

import json
import os

import pandas as pd

RECAPS_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "recaps.json")

# Same deterministic-phrasing-pool pattern as report/build_report.py's
# OPENERS/TRANSITIONS -- picked by a stable hash of the week so a re-run
# grading the same rows again produces the same recap, not a shuffled one.
BIG_CALL_OPENERS = [
    "The model's best call of the week: ",
    "Sharpest read of the week: ",
    "The clearest win this week: ",
]
MISS_OPENERS = [
    "The one that didn't work out: ",
    "Where the model missed: ",
    "The tougher result to swallow: ",
]


def _pick(pool: list, seed: int) -> str:
    return pool[seed % len(pool)]


def _week_seed(season: int, week: int) -> int:
    import hashlib
    return int(hashlib.md5(f"{season}-{week}-recap".encode()).hexdigest(), 16)


def _game_sentences(newly_graded_games: pd.DataFrame, seed: int) -> list[str]:
    graded = newly_graded_games[newly_graded_games["correct"].notna()].copy()
    if graded.empty:
        return []
    # logs/season_results.csv is a CSV round trip (run_week.py's own
    # LOG_PATH) with pending rows mixed in as blanks -- pandas reads a
    # "correct" column like that as object dtype, not bool, even after
    # filtering to just the graded rows. `~graded["correct"]` on an
    # object-dtype column of Python bools does bitwise-NOT on the
    # underlying int (~True == -2), not logical negation, and blows up
    # downstream indexing -- caught testing this against a real
    # CSV-shaped mix of graded/pending rows, not an all-clean synthetic
    # frame. Cast explicitly before any `~` use.
    graded["correct"] = graded["correct"].astype(bool)

    sentences = []
    n_correct = int(graded["correct"].sum())
    n_total = len(graded)
    sentences.append(f"Straight-up, the model went {n_correct}-{n_total - n_correct} on graded games this week.")

    beat_market = graded["model_beat_market"].dropna()
    if not beat_market.empty:
        n_beat = int(beat_market.sum())
        sentences.append(f"Against the spread, it beat the market's own line on {n_beat} of the "
                          f"{len(beat_market)} games where the two actually disagreed.")

    # Biggest correct call: right, and the model disagreed with the
    # market by the widest margin -- the boldest call that actually paid off.
    correct_with_edge = graded[graded["correct"] & graded["edge"].notna() & (graded["edge"].abs() > 1)]
    if not correct_with_edge.empty:
        best = correct_with_edge.loc[correct_with_edge["edge"].abs().idxmax()]
        sentences.append(
            f"{_pick(BIG_CALL_OPENERS, seed)}{best['predicted_winner']} over "
            f"{best['away_team'] if best['predicted_winner'] == best['home_team'] else best['home_team']}, "
            f"a {abs(best['edge']):.1f}-point disagreement with the market that came in right."
        )

    # Biggest miss: wrong, at the highest model confidence -- the pick
    # the model was most sure of that didn't happen.
    wrong = graded[~graded["correct"]].copy()
    if not wrong.empty:
        wrong["confidence"] = wrong["home_win_prob"].apply(lambda p: max(p, 1 - p))
        worst = wrong.loc[wrong["confidence"].idxmax()]
        loser = worst["home_team"] if worst["predicted_winner"] == worst["away_team"] else worst["away_team"]
        sentences.append(
            f"{_pick(MISS_OPENERS, seed)}{worst['predicted_winner']} was favored at "
            f"{worst['confidence'] * 100:.0f}% over {loser}, and {loser} won instead."
        )

    return sentences


def _prop_sentences(newly_graded_props: pd.DataFrame, seed: int) -> list[str]:
    graded = newly_graded_props[newly_graded_props["model_correct"].notna()].copy()
    if graded.empty:
        return []
    graded["model_correct"] = graded["model_correct"].astype(bool)  # see _game_sentences' comment on why

    n_correct = int(graded["model_correct"].sum())
    n_total = len(graded)
    sentences = [f"On player props, the model hit {n_correct} of {n_total} graded picks."]

    correct_with_edge = graded[graded["model_correct"] & graded["edge"].notna()]
    if not correct_with_edge.empty:
        best = correct_with_edge.loc[correct_with_edge["edge"].abs().idxmax()]
        side = "scored" if best["actual_over"] else "didn't score"
        # Full name again, not a "last name" extracted via split()[-1] --
        # that grabs a suffix ("III", "Jr.") for plenty of real players
        # (caught testing this against "Kenneth Walker III" specifically,
        # which produced "and III scored" instead of "and Walker scored").
        sentences.append(f"Best prop call: {best['player']} anytime TD, where the model's own number was "
                          f"furthest from the market's and {best['player']} {side}, as projected.")

    return sentences


def build_recap(newly_graded_games: pd.DataFrame, newly_graded_props: pd.DataFrame,
                 season: int, week: int) -> str | None:
    """None if nothing was actually graded this run (mid-week, nothing
    finished yet) -- never a recap about zero real outcomes."""
    seed = _week_seed(season, week)
    sentences = _game_sentences(newly_graded_games, seed) + _prop_sentences(newly_graded_props, seed)
    if not sentences:
        return None
    return " ".join(sentences)


def save_recap(season: int, week: int, text: str) -> None:
    recaps = load_recaps()
    recaps = [r for r in recaps if not (r["season"] == season and r["week"] == week)]
    recaps.append({"season": season, "week": week, "text": text})
    os.makedirs(os.path.dirname(RECAPS_PATH), exist_ok=True)
    with open(RECAPS_PATH, "w") as f:
        json.dump(recaps, f, indent=2)


def load_recaps() -> list[dict]:
    if not os.path.exists(RECAPS_PATH):
        return []
    with open(RECAPS_PATH) as f:
        return json.load(f)


def latest_recap() -> dict | None:
    recaps = load_recaps()
    if not recaps:
        return None
    return max(recaps, key=lambda r: (r["season"], r["week"]))


RECAP_BLOCK = """<div class="chart-card recap-card">
  <div class="chart-title">Week {week} Recap</div>
  <p class="recap-text">{text}</p>
</div>"""

RECAP_STYLE = """
.recap-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px 18px; }
.recap-text { margin: 0; font-size: 14.5px; line-height: 1.6; color: var(--ink); }
"""


def recap_html() -> str:
    recap = latest_recap()
    if not recap:
        return ""
    return RECAP_BLOCK.format(week=recap["week"], text=recap["text"])
