# tests for data/odds_aggregation.py's multi-book consensus math
"""american_to_prob/devig_pair are checked against known, hand-verified
American-odds conversions; _consensus_point's tie-break (lower-middle,
not statistical median) gets its own explicit test since the module's
own docstring calls out a specific case (median(44.0, 44.5) = 44.25)
that would silently synthesize a line no book ever quoted."""

import pytest

from data.odds_aggregation import (
    _consensus_point, aggregate_one_sided, aggregate_two_sided, american_to_prob, devig_pair, ladder_by_point,
)


def test_american_to_prob_known_values():
    # -110 (standard vig line) -> 110/210
    assert american_to_prob(-110) == pytest.approx(110 / 210, abs=1e-9)
    # +100 (even money) -> exactly 0.5
    assert american_to_prob(100) == pytest.approx(0.5, abs=1e-9)
    # +150 -> 100/250 = 0.4
    assert american_to_prob(150) == pytest.approx(0.4, abs=1e-9)
    # -200 (heavy favorite) -> 200/300
    assert american_to_prob(-200) == pytest.approx(200 / 300, abs=1e-9)


def test_devig_pair_normalizes_to_one():
    prob_a, prob_b = devig_pair(-110, -110)
    assert prob_a == pytest.approx(0.5, abs=1e-9)
    assert prob_b == pytest.approx(0.5, abs=1e-9)
    assert prob_a + prob_b == pytest.approx(1.0, abs=1e-9)


def test_devig_pair_removes_real_overround():
    # -110/-110 each imply ~52.4% (110/210) -- summing to ~104.8%, the
    # book's vig. After de-vig, the pair should land exactly on 50/50
    # for a symmetric line.
    prob_a, prob_b = devig_pair(-110, -110)
    raw_a = american_to_prob(-110)
    assert raw_a > 0.5  # confirms there WAS real vig to remove
    assert prob_a == pytest.approx(0.5, abs=1e-9)


def test_consensus_point_picks_the_mode():
    assert _consensus_point([3.5, 3.5, 3.5, 4.0]) == 3.5


def test_consensus_point_even_tie_picks_lower_middle_not_median():
    """The exact case the module's own docstring warns about: a plain
    median of a tied set can synthesize a point (44.25) that doesn't
    match any book's actual quote. Lower-middle of the tied values must
    be one of the values that was actually quoted."""
    tied_points = [44.0, 44.0, 44.5, 44.5]  # two books each on 44.0 and 44.5
    result = _consensus_point(tied_points)
    assert result in tied_points
    assert result == 44.0  # lower-middle of a 2-element tied set


def test_consensus_point_odd_tie_picks_true_middle():
    # three-way tie -> lower-middle of [1.0, 2.0, 3.0] is the middle element, 2.0
    assert _consensus_point([1.0, 2.0, 3.0]) == 2.0


def test_aggregate_two_sided_empty_returns_none():
    assert aggregate_two_sided([]) is None


def test_aggregate_two_sided_averages_only_books_at_consensus_point():
    quotes = [
        {"book": "a", "point": 3.5, "price_a": -110, "price_b": -110},
        {"book": "b", "point": 3.5, "price_a": -105, "price_b": -115},
        # A book quoting a materially different POINT -- must be excluded
        # from the average entirely, per the module's own stated contract
        # ("a different bet, not noise around the same one").
        {"book": "c", "point": 7.0, "price_a": -200, "price_b": 170},
    ]
    result = aggregate_two_sided(quotes)
    assert result["point"] == 3.5
    assert result["n_books"] == 2  # only a and b, not c


def test_aggregate_one_sided_empty_returns_none():
    assert aggregate_one_sided([]) is None


def test_aggregate_one_sided_averages_raw_implied_probability():
    result = aggregate_one_sided([-110, +120])
    expected = (american_to_prob(-110) + american_to_prob(120)) / 2
    assert result["prob"] == pytest.approx(expected, abs=1e-9)
    assert result["n_books"] == 2


def test_ladder_by_point_keeps_single_book_rungs_with_n_books_1():
    """Single-book points are still included (not silently dropped) --
    the module's own docstring is explicit that excluding them would make
    a real, currently-live line vanish; n_books just tells the caller how
    much to trust each rung."""
    quotes = [
        {"point": 40.5, "price_a": -110, "price_b": -110},
        {"point": 41.5, "price_a": -110, "price_b": -110},
    ]
    ladder = ladder_by_point(quotes)
    assert len(ladder) == 2
    assert all(rung["n_books"] == 1 for rung in ladder)


def test_ladder_by_point_sorted_ascending_by_point():
    quotes = [
        {"point": 45.5, "price_a": -110, "price_b": -110},
        {"point": 38.5, "price_a": -110, "price_b": -110},
        {"point": 42.0, "price_a": -110, "price_b": -110},
    ]
    ladder = ladder_by_point(quotes)
    assert [r["point"] for r in ladder] == [38.5, 42.0, 45.5]
