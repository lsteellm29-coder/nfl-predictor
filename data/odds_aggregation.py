# combines the same market across multiple sportsbooks into one number
"""Multi-book aggregation, shared by data/fetch_week.py's game lines and
data/fetch_props.py's player props. Both used to pick a single book (by a
priority list) for each market -- one book's own quirks (a stale line, a
fat-fingered price) then became the model's entire read on "the market,"
even when several other books were quoting something different that
week. This takes the line most books agree on (the common case) and
averages the de-vigged probability across every book quoting that exact
line, rather than picking or ignoring a single book's number outright.

A book that disagrees on the LINE itself (not just the price) isn't
averaged in -- a book quoting a materially different point is a different
bet, not noise around the same one, so folding its price into an average
alongside books at the real market line would blur two different
numbers together instead of reducing noise on one.
"""

import statistics
from collections import Counter


def american_to_prob(price: float) -> float:
    return 100 / (price + 100) if price > 0 else -price / (-price + 100)


def devig_pair(price_a: float, price_b: float) -> tuple[float, float]:
    """Removes a single book's overround by normalizing both sides'
    implied probabilities to sum to 1."""
    prob_a, prob_b = american_to_prob(price_a), american_to_prob(price_b)
    total = prob_a + prob_b
    return prob_a / total, prob_b / total


def _consensus_point(points: list[float]) -> float:
    """Whichever point value the most books agree on; ties broken
    deterministically by picking the lower-middle of the tied values
    (not their statistical median -- for an even-length tie that can
    synthesize a number that was never actually quoted by any book, e.g.
    median(44.0, 44.5) = 44.25, which then matches zero quotes)."""
    counts = Counter(points)
    max_count = max(counts.values())
    tied = sorted(p for p, c in counts.items() if c == max_count)
    return tied[(len(tied) - 1) // 2]


def aggregate_two_sided(quotes: list[dict]) -> dict | None:
    """quotes: [{"book": key, "point": float, "price_a": int, "price_b": int}, ...]
    for one market (one player-or-game, one stat) across every book
    quoting both sides -- "a" and "b" are the two sides in whatever order
    the caller cares about (over/under, home/away). Returns {"point":
    consensus line, "prob_a": ..., "prob_b": ..., "price_a": ..., "price_b":
    ..., "n_books": how many books that consensus line is averaged
    across}, or None if `quotes` is empty. price_a/price_b in the result
    are each side's median price among books at the consensus point --
    representative of what's actually being offered, not just an
    average-of-two-different-numbers artifact."""
    if not quotes:
        return None
    point = _consensus_point([q["point"] for q in quotes])
    at_point = [q for q in quotes if q["point"] == point]

    probs_a, probs_b = [], []
    for q in at_point:
        pa, pb = devig_pair(q["price_a"], q["price_b"])
        probs_a.append(pa)
        probs_b.append(pb)

    return {
        "point": point,
        "prob_a": statistics.mean(probs_a),
        "prob_b": statistics.mean(probs_b),
        "price_a": statistics.median(q["price_a"] for q in at_point),
        "price_b": statistics.median(q["price_b"] for q in at_point),
        "n_books": len(at_point),
    }


def aggregate_one_sided(prices: list[float]) -> dict | None:
    """prices: raw American odds for one side of a market from every book
    quoting it, with no complementary side to de-vig against (e.g.
    anytime-TD "Yes", where most books don't post a "No" leg). Averages
    each book's own raw implied probability -- not de-vigged (there's
    nothing to de-vig against), so this carries whatever overround each
    book has baked in, same limitation the single-book version already
    had."""
    if not prices:
        return None
    probs = [american_to_prob(p) for p in prices]
    return {
        "prob": statistics.mean(probs),
        "price": statistics.median(prices),
        "n_books": len(prices),
    }


def ladder_by_point(quotes: list[dict]) -> list[dict]:
    """quotes: same shape as aggregate_two_sided's input, but for a market
    that's meant to have MANY point values (alternate spreads/totals),
    not one consensus line. Groups by point and runs aggregate_two_sided
    within each group, so each rung of the ladder gets its own
    multi-book-averaged probability instead of collapsing the whole
    market down to a single number. Points with only one book quoting
    them are still included (n_books=1) -- excluding them would silently
    make a real, currently-live line vanish -- but n_books is always
    returned so a caller can choose to only trust points multiple books
    agree exist."""
    by_point: dict[float, list[dict]] = {}
    for q in quotes:
        by_point.setdefault(q["point"], []).append(q)
    rungs = [aggregate_two_sided(group) for group in by_point.values()]
    return sorted(rungs, key=lambda r: r["point"])
