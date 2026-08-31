# Week 1 Audit & Tuning Plan Phase 4.2: fit roster-adjustment weights against real outcomes
"""Builds the "roster change" signal the plan describes (QB change,
O-line continuity, coaching change, skill-position turnover, defensive
front-seven turnover) as a standalone, isolated feature set -- computed
from data already fetched/validated elsewhere in this codebase
(data/team_change_tracker.py's per-unit snap-share turnover and coaching-
change detection, both already season-parameterized and reused
unmodified here; QB change is new, detected directly off the schedule's
own home_qb_id/away_qb_id columns) -- then fits it with ridge regression
against real historical Week 1 margins, 2017-2025 (2016 excluded: no
2015 cache to compare against, same "honest data limit" as every other
prior-season fallback in this project).

This is NOT wired into model/train.py's FEATURE_COLS -- the plan's own
Section 4.2 asks for a "separate, isolated module," and
data/team_change_tracker.py's docstring already commits to staying
narrative-only. This script answers a different, standalone question:
if you DID want to weight these five signals, what would real Week 1
outcomes say the weights should be -- and are any of them actually
distinguishable from zero.

Confidence intervals via bootstrap resampling (not a closed-form OLS CI
-- ridge regression doesn't have one), since the plan explicitly asks
to zero out any weight whose CI crosses zero.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from config import HISTORICAL_SEASONS
from data.team_change_tracker import head_coach_changes, team_unit_turnover

RIDGE_ALPHA = 1.0
N_BOOTSTRAP = 1000
CI_PERCENTILES = (2.5, 97.5)

# Weeks 1 the roster-change signal covers: needs a prior season to
# compare against, same limit as every other prior-season fallback in
# this codebase.
FIT_SEASONS = HISTORICAL_SEASONS[1:]


def _team_qb_at_week(schedules: pd.DataFrame, season: int, team: str, week) -> str | None:
    """That team's QB in a specific (season, week) game, home or away
    side, whichever they were on. None if no such game is in the cache
    (bye week, or the season's own final week already passed)."""
    games = schedules[
        (schedules["season"] == season)
        & ((schedules["home_team"] == team) | (schedules["away_team"] == team))
    ]
    if week == "last":
        games = games[games["game_type"] == "REG"].sort_values("week")
        if games.empty:
            return None
        row = games.iloc[-1]
    else:
        row = games[games["week"] == week]
        if row.empty:
            return None
        row = row.iloc[0]
    return row["home_qb_id"] if row["home_team"] == team else row["away_qb_id"]


def _qb_changes_for_season(schedules: pd.DataFrame, season: int, teams: list[str]) -> dict[str, bool]:
    """team -> True if that team's Week 1 (season) starting QB differs
    from their own final game of season-1 -- the same "final row of the
    prior season" comparison point every other fallback in this project
    uses, for consistency."""
    out = {}
    for team in teams:
        prior_qb = _team_qb_at_week(schedules, season - 1, team, "last")
        current_qb = _team_qb_at_week(schedules, season, team, 1)
        if prior_qb is None or current_qb is None:
            continue
        out[team] = prior_qb != current_qb
    return out


def _front_seven_turnover(unit_turnover: dict) -> dict | None:
    """Pools data/team_change_tracker.py's separate "defensive front" and
    "linebackers" units into one combined front-seven turnover_pct --
    summing departed/total counts (not averaging the two units'
    percentages, which would wrongly weight a 3-player LB corps the same
    as an 8-player DL rotation)."""
    front = unit_turnover.get("defensive front")
    lb = unit_turnover.get("linebackers")
    if not front and not lb:
        return None
    departed = (front["departed"] if front else 0) + (lb["departed"] if lb else 0)
    total = (front["total"] if front else 0) + (lb["total"] if lb else 0)
    if total == 0:
        return None
    return {"turnover_pct": departed / total, "departed": departed, "total": total}


def build_roster_change_dataset(schedules: pd.DataFrame) -> pd.DataFrame:
    """One row per Week 1 game, `{signal}_diff` = home - away, same
    diffing convention model/train.py's own FEATURE_COLS already uses,
    plus the actual point margin as the fit target."""
    rows = []
    for season in FIT_SEASONS:
        week1 = schedules[
            (schedules["season"] == season) & (schedules["week"] == 1)
            & (schedules["game_type"] == "REG") & schedules["home_score"].notna()
        ]
        if week1.empty:
            continue
        teams = sorted(set(week1["home_team"]) | set(week1["away_team"]))
        try:
            unit_turnover = team_unit_turnover(season)
            coach_changes = head_coach_changes(season)
        except Exception as e:
            print(f"  skipping {season}: couldn't compute team-change signals ({e})")
            continue
        qb_changes = _qb_changes_for_season(schedules, season, teams)

        for _, g in week1.iterrows():
            home, away = g["home_team"], g["away_team"]
            if home not in qb_changes or away not in qb_changes:
                continue  # missing QB history for one side -- skip rather than guess

            def signal(team):
                unit = unit_turnover.get(team, {})
                ol = unit.get("offensive line", {}).get("turnover_pct")
                skill = unit.get("offensive skill", {}).get("turnover_pct")
                front7 = _front_seven_turnover(unit)
                return {
                    "qb_change": float(qb_changes[team]),
                    "ol_turnover": ol if ol is not None else 0.0,
                    "coaching_change": float(team in coach_changes),
                    "skill_turnover": skill if skill is not None else 0.0,
                    "front7_turnover": front7["turnover_pct"] if front7 else 0.0,
                }

            home_sig, away_sig = signal(home), signal(away)
            rows.append({
                "season": season, "home_team": home, "away_team": away,
                "margin": g["home_score"] - g["away_score"],
                **{f"{k}_diff": home_sig[k] - away_sig[k] for k in home_sig},
            })
    return pd.DataFrame(rows)


SIGNAL_COLS = ["qb_change_diff", "ol_turnover_diff", "coaching_change_diff",
               "skill_turnover_diff", "front7_turnover_diff"]


def fit_with_bootstrap_ci(data: pd.DataFrame) -> pd.DataFrame:
    X, y = data[SIGNAL_COLS].values, data["margin"].values
    point = Ridge(alpha=RIDGE_ALPHA).fit(X, y)

    rng = np.random.default_rng(42)
    boot_coefs = np.zeros((N_BOOTSTRAP, len(SIGNAL_COLS)))
    n = len(data)
    for i in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        boot_coefs[i] = Ridge(alpha=RIDGE_ALPHA).fit(X[idx], y[idx]).coef_

    lo, hi = np.percentile(boot_coefs, CI_PERCENTILES, axis=0)
    rows = []
    for i, col in enumerate(SIGNAL_COLS):
        crosses_zero = lo[i] <= 0 <= hi[i]
        rows.append({
            "signal": col, "weight": point.coef_[i],
            "ci_low": lo[i], "ci_high": hi[i],
            "final_weight": 0.0 if crosses_zero else point.coef_[i],
            "zeroed": crosses_zero,
        })
    return pd.DataFrame(rows)


def main():
    schedules = pd.read_parquet("data/cache/schedules.parquet")
    print(f"Building roster-change dataset for Week 1, seasons {FIT_SEASONS[0]}-{FIT_SEASONS[-1]}...")
    data = build_roster_change_dataset(schedules)
    print(f"{len(data)} Week 1 games with complete roster-change data.\n")

    result = fit_with_bootstrap_ci(data)
    print(f"{'signal':22s} {'weight':>8s} {'95% CI':>20s} {'final':>8s}")
    for _, r in result.iterrows():
        ci = f"[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]"
        flag = " (zeroed -- CI crosses 0)" if r["zeroed"] else ""
        print(f"{r['signal']:22s} {r['weight']:+8.2f} {ci:>20s} {r['final_weight']:+8.2f}{flag}")
    return data, result


if __name__ == "__main__":
    main()
