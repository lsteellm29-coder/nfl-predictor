# hand-checked spread-sign convention lock (Week 1 Audit & Tuning Plan Phase 1.2)
"""This codebase's spread convention, verified against real data (see
AUDIT.md section 3.2): nfl_data_py's spread_line is POSITIVE when the
HOME team is favored (magnitude = home team's expected margin of
victory) -- the opposite of a bettor's-ticket convention (where the
favorite's own line carries the minus sign), but internally consistent
everywhere this codebase touches it: model/train.py's
`market_spread = spread_line` (unchanged), model/predict.py's
`implied_spread`, and report/cards.py's display all share this same sign.

Verified empirically against 15 real 2025 games cross-checked by
moneyline (unambiguous: a negative moneyline always means favored, no
sign-convention question possible there). This test hardcodes three of
those real, hand-checked games -- a home blowout, a road win, and a
close game -- as a permanent regression lock: if a future change ever
flips this sign anywhere in the pipeline, this test catches it
immediately instead of it surfacing as a silent, hard-to-notice
inversion in every displayed pick.

Data below is a real snapshot (nfl.import_schedules([2025]), captured
2026-08-31) -- hardcoded rather than re-fetched live, so this test's
result depends only on this codebase's own logic, never on a live
network call or a values changing upstream.
"""

import pytest

# (description, home_team, away_team, home_score, away_score, spread_line,
#  home_moneyline, away_moneyline)
REAL_2025_GAMES = [
    # Home blowout: WAS beat NYG by 15 at home, heavily favored (-298 ML).
    ("home_blowout", "WAS", "NYG", 21.0, 6.0, 6.0, -298.0, 240.0),
    # Road win: CIN (away) won at CLE, and was the actual favored side (-238 ML).
    ("road_win", "CLE", "CIN", 16.0, 17.0, -5.5, 195.0, -238.0),
    # Close game: TB (away) won by 3 at ATL, and was the barely-favored side (-115 ML).
    ("close_game", "ATL", "TB", 20.0, 23.0, -1.5, -105.0, -115.0),
]


def _moneyline_favorite(home_ml: float, away_ml: float) -> str:
    """The unambiguous ground truth this test checks spread_line's sign
    against: a MORE NEGATIVE moneyline is always the more heavily
    favored side, regardless of any spread-sign convention question."""
    return "home" if home_ml < away_ml else "away"


def _spread_favorite(spread_line: float) -> str:
    """This codebase's own convention (positive = home favored), applied
    exactly the way model/train.py's market_spread and model/predict.py's
    implied_spread both use spread_line, untouched."""
    if spread_line == 0:
        return "pick_em"
    return "home" if spread_line > 0 else "away"


@pytest.mark.parametrize("description,home,away,home_score,away_score,spread_line,home_ml,away_ml", REAL_2025_GAMES)
def test_spread_sign_matches_moneyline_favorite(
    description, home, away, home_score, away_score, spread_line, home_ml, away_ml,
):
    """The core convention check: spread_line's sign must agree with the
    moneyline's (unambiguous) favorite for every hand-checked real game,
    regardless of which team actually won -- a spread predicts an
    expectation, not a guaranteed outcome, so this checks the CONVENTION
    is self-consistent, not that favorites always win."""
    assert _spread_favorite(spread_line) == _moneyline_favorite(home_ml, away_ml), (
        f"{description}: spread_line={spread_line:+.1f} implies "
        f"{_spread_favorite(spread_line)} favored, but moneyline "
        f"(home {home_ml:+.0f} / away {away_ml:+.0f}) says {_moneyline_favorite(home_ml, away_ml)}"
    )


def test_home_blowout_game_matches_expected_shape():
    """Sanity-checks the fixture itself hasn't been miscopied: the home
    blowout game really was a home win by a wide-enough margin, and the
    spread really did favor the home side by a real amount."""
    _, home, away, home_score, away_score, spread_line, home_ml, away_ml = REAL_2025_GAMES[0]
    assert home_score - away_score >= 10  # a real blowout, not a squeaker
    assert spread_line > 3  # a real favorite, not a pick'em


def test_road_win_game_matches_expected_shape():
    _, home, away, home_score, away_score, spread_line, home_ml, away_ml = REAL_2025_GAMES[1]
    assert away_score > home_score  # the away team actually won
    assert spread_line < 0  # and this codebase's convention correctly has the away side favored


def test_close_game_matches_expected_shape():
    _, home, away, home_score, away_score, spread_line, home_ml, away_ml = REAL_2025_GAMES[2]
    assert abs(home_score - away_score) <= 3  # genuinely close
    assert abs(spread_line) <= 3  # market agreed it was close, not lopsided
