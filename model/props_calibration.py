# checks whether player-prop confidence actually means what it says, per stat category
"""Player-props reliability check (Audit Fix Plan Step 3) -- the same
question model/calibration.py asks for game picks ("when the model says
70%, does it actually hit 70% of the time?"), applied to logs/props_results.csv.

Run separately per stat category (yardage: pass/rush/rec_yards; reception
count: receptions; TD: anytime_td) rather than pooled, since a normal-
distribution yardage prop and a Poisson TD prop are different bet shapes
with no reason to share a calibration curve -- pooling them would let one
category's miscalibration hide inside another's average.

Unlike model/calibration.py, this has no historical backtest to fall back
on: The Odds API only exposes live/current props, there's no archived
historical player-props line dataset to grade against, so this can only
ever be as good as the real graded weeks accumulated in
logs/props_results.csv via run_week.py's log_props_week(). Early in a
season (or right after this was first wired in) there may not be enough
graded props yet for a bucket to mean anything -- this reports that
honestly (a "not enough data" bucket) rather than drawing a reliability
curve out of a handful of games.
"""

import os

import pandas as pd

MIN_GRADED_FOR_BUCKET = 8  # below this, a hit rate is a coin flip away from meaningless

STAT_CATEGORY = {
    "pass_yards": "yardage", "rush_yards": "yardage", "rec_yards": "yardage",
    "receptions": "reception_count", "anytime_td": "td",
}

BUCKET_EDGES = [0.0, 0.5, 0.55, 0.6, 0.65, 0.7, 1.0]

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROPS_LOG_PATH = os.path.join(ROOT_DIR, "logs", "props_results.csv")


def _confidence_and_correctness(df: pd.DataFrame) -> pd.DataFrame:
    """Reframes each graded prop from (model_over_prob, actual_over) into
    (confidence, hit) -- confidence is how strongly the model leaned
    toward whichever side it actually picked (always >=0.5, symmetric
    around a coin flip), hit is whether that side was right. Same
    reframing model/calibration.py's reliability_table() implicitly gets
    for free from a single win-probability side (home); props need it
    explicit since "over" isn't inherently the side being claimed."""
    picked_over = df["model_over_prob"] >= 0.5
    confidence = df["model_over_prob"].where(picked_over, 1 - df["model_over_prob"])
    hit = picked_over == df["actual_over"]
    return pd.DataFrame({"confidence": confidence, "hit": hit})


def reliability_table(graded: pd.DataFrame) -> pd.DataFrame:
    """graded: rows from logs/props_results.csv with a non-null
    actual_value (already played and graded). Buckets by the model's own
    confidence in whichever side it picked, same predicted-vs-actual
    shape as model/calibration.py's reliability_table."""
    reframed = _confidence_and_correctness(graded)
    reframed["bucket"] = pd.cut(reframed["confidence"], BUCKET_EDGES)
    return reframed.groupby("bucket", observed=True).agg(
        n=("hit", "size"), predicted_mean=("confidence", "mean"), actual_hit_rate=("hit", "mean"),
    )


def category_reliability(graded: pd.DataFrame) -> dict:
    """{category: reliability_table} for every stat category with at
    least one graded row -- yardage/reception_count/td kept fully
    separate per this module's docstring."""
    graded = graded.copy()
    graded["category"] = graded["stat"].map(STAT_CATEGORY)
    return {
        category: reliability_table(sub)
        for category, sub in graded.groupby("category")
    }


def market_comparison(graded: pd.DataFrame) -> dict:
    """Overall hit rate for the model's own pick vs. the market's favored
    side vs. how often the model beat the market when they disagreed --
    same three-number summary model/calibration.py's Brier score serves
    for game picks, just in hit-rate terms since props are graded
    per-pick rather than as a single probability distribution."""
    disagreed = graded[graded["model_beat_market"].notna()]
    return {
        "n_graded": len(graded),
        "model_hit_rate": float(graded["model_correct"].mean()) if len(graded) else None,
        "market_hit_rate": float(graded["market_correct"].mean()) if len(graded) else None,
        "n_disagreed": len(disagreed),
        "model_beat_market_rate": float(disagreed["model_beat_market"].mean()) if len(disagreed) else None,
    }


def main():
    if not os.path.exists(PROPS_LOG_PATH) or os.path.getsize(PROPS_LOG_PATH) == 0:
        print(f"No props log found at {PROPS_LOG_PATH} yet -- nothing to check. "
              f"This fills in as run_week.py logs and grades real weeks.")
        return

    log_df = pd.read_csv(PROPS_LOG_PATH)
    graded = log_df[log_df["actual_value"].notna()].copy()
    graded["actual_over"] = graded["actual_over"].astype(bool)
    graded["model_correct"] = graded["model_correct"].astype(bool)
    graded["market_correct"] = graded["market_correct"].astype(bool)

    print(f"Props log: {len(log_df)} logged, {len(graded)} graded (games played), "
          f"{len(log_df) - len(graded)} still pending.")
    if graded.empty:
        print("No graded props yet -- nothing to calibrate against until at least one "
              "logged week's games have been played.")
        return

    summary = market_comparison(graded)
    print(f"\nOverall: model {summary['model_hit_rate']:.1%} vs market {summary['market_hit_rate']:.1%} "
          f"hit rate ({summary['n_graded']} graded props).")
    if summary["n_disagreed"]:
        print(f"When model and market disagreed ({summary['n_disagreed']} props): "
              f"model was right {summary['model_beat_market_rate']:.1%} of the time.")
    else:
        print("Model and market never disagreed on a side across the graded sample yet.")

    for category, table in category_reliability(graded).items():
        n = table["n"].sum()
        print(f"\n{category} ({n} graded):")
        if n < MIN_GRADED_FOR_BUCKET:
            print(f"  Not enough graded props yet (<{MIN_GRADED_FOR_BUCKET}) for a bucketed "
                  f"reliability table to mean anything.")
            continue
        print(table)


if __name__ == "__main__":
    main()
