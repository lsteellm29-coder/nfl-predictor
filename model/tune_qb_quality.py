# Post-freeze accuracy experiment: QB quality delta instead of a binary change flag
"""TUNING.md's Phase 4.2 section found `qb_change_diff` (a binary "did the
starting QB change" flag) was NOT distinguishable from zero (weight
-0.06, CI [-3.27, +3.35]) and proposed the likely explanation: the flag
can't tell an upgrade (benching a struggler for a good rookie, landing a
proven veteran) from a downgrade (a good starter's backup stepping in),
so the two cancel out. The proposed follow-up was a continuous "QB
quality delta" instead of the binary flag, using the per-QB EPA game
logs data/player_trends.py already builds -- this is that follow-up.

Quality = a QB's own PRIOR-SEASON average EPA/dropback (wherever they
played it -- a real, knowable-before-kickoff number, same "only data
knowable before the game" discipline as everything else in this
project), requiring at least MIN_DROPBACKS_FOR_QUALITY to trust that
season's average as real signal rather than small-sample noise. A Week
1 starter with no qualifying prior-season record (a rookie, or a
backup who barely played) has no knowable quality number -- skipped,
not guessed at, same "skip rather than guess" rule
model/tune_roster_weights.py already applies to missing QB/coach
history.

Two tests, same 2017-2025 Week 1 games as tune_roster_weights.py:
1. qb_quality_diff on its own (does the raw signal relate to margin at
   all).
2. qb_quality_diff swapped in for qb_change_diff alongside the other
   four already-fitted roster signals -- the direct, apples-to-apples
   test of whether continuous quality does what the binary flag
   couldn't, on the exact same sample tune_roster_weights.py used.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from model.tune_roster_weights import CI_PERCENTILES, FIT_SEASONS, N_BOOTSTRAP, RIDGE_ALPHA, build_roster_change_dataset

# Roughly 4-6 full games' worth -- below this, a season's average EPA/
# dropback is too small a sample to call "quality" rather than noise
# (data/player_trends.py's own hot/cold streak detection uses a similar
# minimum-games floor for the same reason).
MIN_DROPBACKS_FOR_QUALITY = 100


def _qb_season_epa(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, passer_player_id): that QB's average EPA per
    dropback across every team they played for that season, and the
    dropback count it's based on."""
    dropbacks = pbp[(pbp["qb_dropback"] == 1) & pbp["passer_player_id"].notna()]
    return (
        dropbacks.groupby(["season", "passer_player_id"])
        .agg(epa=("epa", "mean"), dropbacks=("epa", "size"))
        .reset_index()
    )


def _qb_quality(qb_season_epa: pd.DataFrame, season: int, qb_id) -> float | None:
    if pd.isna(qb_id):
        return None
    row = qb_season_epa[(qb_season_epa["season"] == season) & (qb_season_epa["passer_player_id"] == qb_id)]
    if row.empty or row.iloc[0]["dropbacks"] < MIN_DROPBACKS_FOR_QUALITY:
        return None
    return float(row.iloc[0]["epa"])


def build_qb_quality_dataset(schedules: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per Week 1 game, `qb_quality_diff` = home - away, using
    each side's own Week 1 starter's PRIOR-season EPA/dropback -- the
    same knowable-before-kickoff discipline every other feature in this
    project follows."""
    qb_season_epa = _qb_season_epa(pbp)
    rows = []
    for season in FIT_SEASONS:
        week1 = schedules[
            (schedules["season"] == season) & (schedules["week"] == 1)
            & (schedules["game_type"] == "REG") & schedules["home_score"].notna()
        ]
        if week1.empty:
            continue
        for _, g in week1.iterrows():
            home_q = _qb_quality(qb_season_epa, season - 1, g.get("home_qb_id"))
            away_q = _qb_quality(qb_season_epa, season - 1, g.get("away_qb_id"))
            if home_q is None or away_q is None:
                continue  # no qualifying prior-season record for one side -- skip, don't guess
            rows.append({
                "season": season, "home_team": g["home_team"], "away_team": g["away_team"],
                "margin": g["home_score"] - g["away_score"],
                "qb_quality_diff": home_q - away_q,
            })
    return pd.DataFrame(rows)


def fit_signal_set(data: pd.DataFrame, signal_cols: list[str]) -> list[dict]:
    """Same ridge + bootstrap-CI machinery as
    tune_roster_weights.fit_with_bootstrap_ci, generalized to take an
    explicit column list -- needed here because this dataset carries the
    swapped-in qb_quality_diff column under a different name than
    tune_roster_weights.SIGNAL_COLS expects, rather than mutating that
    shared module constant."""
    X, y = data[signal_cols].values, data["margin"].values
    point = Ridge(alpha=RIDGE_ALPHA).fit(X, y)

    rng = np.random.default_rng(42)
    boot_coefs = np.zeros((N_BOOTSTRAP, len(signal_cols)))
    n = len(data)
    for i in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        boot_coefs[i] = Ridge(alpha=RIDGE_ALPHA).fit(X[idx], y[idx]).coef_

    lo, hi = np.percentile(boot_coefs, CI_PERCENTILES, axis=0)
    return [
        {"signal": col, "weight": point.coef_[i], "ci_low": lo[i], "ci_high": hi[i], "zeroed": lo[i] <= 0 <= hi[i]}
        for i, col in enumerate(signal_cols)
    ]


def main():
    schedules = pd.read_parquet("data/cache/schedules.parquet")
    pbp = pd.read_parquet(
        "data/cache/pbp.parquet",
        columns=["season", "posteam", "qb_dropback", "passer_player_id", "epa"],
    )

    print("Test 1: qb_quality_diff on its own")
    qb_data = build_qb_quality_dataset(schedules, pbp)
    print(f"{len(qb_data)} Week 1 games with a qualifying prior-season QB record on both sides "
          f"(min {MIN_DROPBACKS_FOR_QUALITY} dropbacks).")
    solo = fit_signal_set(qb_data, ["qb_quality_diff"])[0]
    flag = " (zeroed -- CI crosses 0)" if solo["zeroed"] else " -- CI excludes 0, a real signal"
    print(f"  weight {solo['weight']:+.3f}  95% CI [{solo['ci_low']:+.3f}, {solo['ci_high']:+.3f}]{flag}\n")

    print("Test 2: qb_quality_diff swapped in for qb_change_diff, alongside the other four roster signals")
    roster_data = build_roster_change_dataset(schedules)
    combined = roster_data.merge(
        qb_data[["season", "home_team", "away_team", "qb_quality_diff"]],
        on=["season", "home_team", "away_team"], how="inner",
    )
    print(f"{len(combined)} Week 1 games with complete data for all five signals (QB quality version).")

    swapped_cols = ["qb_quality_diff", "ol_turnover_diff", "coaching_change_diff",
                    "skill_turnover_diff", "front7_turnover_diff"]
    result = fit_signal_set(combined, swapped_cols)
    print(f"{'signal':22s} {'weight':>8s} {'95% CI':>20s} {'zeroed'}")
    for r in result:
        ci = f"[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]"
        print(f"{r['signal']:22s} {r['weight']:+8.2f} {ci:>20s} {r['zeroed']}")

    return solo, result


if __name__ == "__main__":
    main()
