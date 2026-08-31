# SharpLine — Ground Truth Audit (Phase 0)

Produced against commit `8ac651f`. This maps the pipeline as it exists today — nothing here has been fixed yet except where explicitly marked `[FIXED THIS PASS]`. Every claim below was verified against either the source code or a live run of the actual data, not inferred.

---

## 1. Data sources, and where each is loaded

| Source | Library call | Loaded in | Cached to |
|---|---|---|---|
| Historical schedules (2016–2025) | `nfl.import_schedules(HISTORICAL_SEASONS)` | `data/fetch_historical.py` | `data/cache/schedules.parquet` |
| Historical play-by-play (2016–2025) | `nfl.import_pbp_data(HISTORICAL_SEASONS, downcast=True)` | `data/fetch_historical.py` | `data/cache/pbp.parquet` |
| Historical injury reports | `nfl.import_injuries(seasons)` | `data/fetch_injuries.py: historical_injury_impact()` | not cached — recomputed at train time |
| Team-game / rolling stats | derived from schedules + pbp | `data/team_stats.py` | `data/cache/team_stats.parquet` |
| Live current-season schedule | `nfl.import_schedules([CURRENT_SEASON])` | `model/predict.py`, `run_week.py`, `data/team_change_tracker.py`, `report/logos.py`, `data/fetch_week.py` (≥8 call sites) | not cached — live every run |
| Live current-season pbp | `nfl.import_pbp_data([CURRENT_SEASON], downcast=True)` | `model/predict.py`, `model/player_stats.py` | not cached |
| Seasonal rosters | `nfl.import_seasonal_rosters(seasons)` | routed through `data/rosters.py: fetch_rosters()` (only remaining raw calls: `qa/validate_rosters.py`'s two functions, deliberately, see §4 below) | not cached (except `data/cache/roster_snapshot.parquet`, a week-over-week diff snapshot) |
| Snap counts | `nfl.import_snap_counts(seasons)` | `model/predict.py` (`_meaningful_usage_players`), `data/team_change_tracker.py` | not cached |
| Live injury status | ESPN unofficial API (`ESPN_INJURIES_URL`) | `data/fetch_injuries.py: fetch_current_player_injury_status()`, `fetch_current_injury_impact()` | `data/cache/injury_history.parquet` (weekly snapshot log, Combined Build Plan Part 4) |
| Live odds/props | The Odds API | `data/fetch_props.py`, `data/odds_aggregation.py` | not cached |
| Live weather forecast | a forecast API | `data/fetch_weather.py` | not cached |
| Live news | ESPN feed + Firecrawl team-beat scraping | `data/fetch_news.py`, `data/fetch_firecrawl_sources.py` | not cached |
| balldontlie (cross-check only) | balldontlie API | `data/fetch_balldontlie.py` | not cached |
| ourlads.com depth charts (cross-check only) | Firecrawl scrape | `data/fetch_firecrawl_sources.py` | not cached |
| Blocked-player list | derived from the two cross-checks above | `qa/validate_rosters.py: save_blocked_players()` | `data/cache/blocked_players.json` |
| Model artifact | trained by `model/train.py` | loaded by `model/predict.py`, `run_week.py`, `build_artifact.py` | `model/model.joblib` |

**Config** (`config.py`): `CURRENT_SEASON = 2026`. `HISTORICAL_SEASONS = range(2016, 2026)` (2016–2025, 10 seasons — note: the tuning plan's Phase 4.1/5 asks for a 2015-anchored backtest; the cache currently only goes back to 2016, one season short of that).

---

## 2. Pipeline shape, stage by stage

**Training path** (`model/train.py: main()`):
1. `schedules.parquet` (all REG+POST+PRE games, 2016–2025) + `team_stats.parquet` (one row per team/season/week, `is_home` flag) →
2. `build_feature_frame()`: filters to `REG` + played + non-tie games, inner-merges home/away team-stats onto the schedule (drops any game where either side has no rolling-stats row — i.e., that team's first game of a season with `min_periods` not yet satisfied), computes 18 `_diff` stat columns + Elo diffs + situational flags → one row per playable game, ~`2200` rows across the 10-season window (verify exact count against a live run, not hardcoded here since it changes as seasons are added)
3. `games.dropna(subset=FEATURE_COLS)` — silently drops any row missing a market spread or any other feature. **Intentional and consistent**: `model/predict.py`'s `_build_features()` applies the identical "no spread_line → can't score this game" rule at inference time, so there's no train/serve skew from this drop — documented in §5 as accepted, not a Phase 1 bug.
4. Two model types trained (logistic, xgboost) on `TRAIN_SEASONS` (all but the last cached season), evaluated on `TEST_SEASON` (last cached season) as the one true untouched holdout. A separate walk-forward loop (`walk_forward_folds()`, already exists) trains on seasons `1..N` and tests `N+1` for every `N ≥ MIN_WALK_FORWARD_TRAIN_SEASONS (6)`, purely to pick which model TYPE to deploy — it does not change what data the final deployed model is fit on.
5. Saved to `model/model.joblib`: both sub-models, `model_type`, `FEATURE_COLS`, a spread-calibration linear regression, and the walk-forward fold results.

**Live scoring path** (`run_week.py` → `model/predict.py: score_week()`):
1. `fetch_week(week, season)` — this week's games from the live schedule.
2. `get_pregame_stats(season, week)`: `_current_season_stats()` (this season's games before `week`, empty if none played) unioned with `_fallback_stats()` (last cached season's final rolling-stat row) for any team with no current-season row yet.
3. `get_current_elo_ratings(season)`: replays the ENTIRE cached historical schedule + current season through `compute_elo_ratings()` every single call (not incremental) — see §3.1, this is where the team-code bug lives.
4. Per game: builds the same `_diff` feature vector `build_feature_frame()` used at training time, scores it through the saved model, computes implied spread via the saved spread-calibration regression, computes `edge = implied_spread - spread_line`.
5. Narrative-lookup layer (positional mismatches, QB/RB streaks, team/coach h2h, team-change tracking) computed once per week, attached per game — never touches the model's own inputs.
6. Confidence-caveat layer (Combined Build Plan Parts 2–3): preseason-fallback + roster-turnover, and coach-change + unit-turnover, both purely display-side.

**Props path** (`model/player_stats.py: score_props()`): separate from the win-probability model entirely — anytime-TD only, own red-zone-touch-share model (`model/td_model.py`), own walk-forward backtest (`model/td_backtest.py`), own calibration audit (`model/td_calibration.py`), own ensemble blend (`model/td_ensemble.py`). Shares `data/rosters.py`'s roster fetch and `data/fetch_injuries.py`'s injury status with the win-probability path, nothing else.

---

## 3. Hardcoded numbers (every one found, with where it lives and whether it's tuned or guessed)

| Constant | Value | File | Tuned or guessed? |
|---|---|---|---|
| `SEASON_REGRESSION` (Elo season-boundary regression) | 1/3 | `model/elo.py` | Guessed — "standard FiveThirtyEight convention," never fit against this project's own data |
| `K_FACTOR` | 20.0 | `model/elo.py` | Guessed |
| `HOME_FIELD_ADV` | 55.0 (Elo points) | `model/elo.py` | Guessed |
| `INITIAL_RATING` | 1500.0 | `model/elo.py` | Standard convention |
| `MIN_WALK_FORWARD_TRAIN_SEASONS` | 6 | `model/train.py` | Reasoned (need enough seasons for a fold to mean anything), not fit |
| XGBoost `max_depth`, `n_estimators`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_lambda` | 3, 200, 0.03, 0.8, 0.8, 1.0 | `model/train.py: train_xgboost()` | Guessed defaults for a small (~2000-row) dataset, not grid-searched |
| `CONFIDENCE_THRESHOLD` (display tier) | 0.60 | `report/cards.py` | Reasoned, not fit |
| `MEANINGFUL_SNAP_PCT` / `MEANINGFUL_MIN_GAMES` ("real starter") | 0.50 / 4 games | `model/predict.py`, `data/team_change_tracker.py` | Reasoned, not fit |
| `NOTABLE_SNAP_PCT` ("genuinely key player") | 0.80 | `data/team_change_tracker.py` | Reasoned, not fit |
| `TURNOVER_CAVEAT_THRESHOLD` / `UNIT_TURNOVER_NOTABLE` | 0.35 / 0.40 | `model/predict.py`, `data/team_change_tracker.py` | Reasoned against qa/validate_rosters.py's own documented 15–25% normal-churn range, not grid-searched |
| `FALLBACK_GAMES_THRESHOLD` | 3 games | `model/predict.py` | Matches this plan's own Phase 3 "no prior-season data" guidance; not fit |
| `COACH_CHANGE_GAMES_WINDOW` | 6 games | `data/team_change_tracker.py` | Reasoned ("roughly a quarter-season"), not fit |
| `RETURN_WINDOW_WEEKS` (cleared-to-play badge) | 2 weeks | `data/fetch_injuries.py` | Reasoned from the spec's own "first game or two back" wording |
| `MIN_ROSTER_SIZE` | 40 | `qa/validate_rosters.py` | Reasoned (real roster is 53+practice squad) |
| `STATUS_WEIGHTS` / `POSITION_WEIGHTS` / `USAGE_MULTIPLIER` (injury impact) | Out=1.0, Doubtful=0.75, Questionable=0.35, Probable=0.1; QB=5.0, skill=2.0, line/front7=1.5, ST=0.75, FB/LS=0.5 | `data/fetch_injuries.py` | **Guessed, never fit or ablation-tested** — this is the single largest unvalidated hardcoded block in the pipeline |
| `MIN_TARGETS` / `MIN_RUSH_ATTEMPTS` (positional-matchup narrative) | WR 40/TE 20/RB 15 targets, 60 rush attempts | `data/positional_matchups.py` | Reasoned, not fit |
| `POSITIONAL_NOTABLE` (narrative-lead threshold) | 0.55 | `report/narrative.py` | **Actually calibrated** — comment states it was set from a real full-season percentile computation (~85th), not guessed |
| `TEAM_H2H_CLEAN_MARGIN_RATE` / `COACH_H2H_CLEAN_MARGIN_RATE` | 0.6 | `report/narrative.py` | Reasoned, not fit |
| `MIN_COACH_MEETINGS` / `MIN_TEAM_MEETINGS` | 3 | `data/team_history.py` | Reasoned, not fit |
| Model-selection tie-break, spread-calibration model | `LinearRegression` on win-prob → market spread | `model/train.py` | This IS the shrinkage-adjacent mechanism this plan's Phase 4.1 is really asking about — see §6 |

**There is no explicit "roster-turnover shrinkage %" constant matching Phase 4.1's framing exactly.** The closest existing analogues are `model/elo.py`'s `SEASON_REGRESSION` (1/3, season-boundary Elo regression) and `model/predict.py`'s `_fallback_stats()` (binary: either use this season's rolling stats, or 100% of last season's — no partial blend). Phase 4.1's grid search is genuinely new work, not re-tuning something that already exists in a different name. Documented for real in §6.

---

## 4. Full path a single game takes, traced end to end (worked example)

Picked `NE @ SEA`, this week's actual opener, spread-checked against a live run:

1. `run_week.py` calls `fetch_week(1, 2026)` → one row from the live schedule fetch, `home_team="SEA"`, `away_team="NE"`, `spread_line` = whatever The Odds API has posted as of the run.
2. `get_pregame_stats(2026, 1)`: 2026 has zero completed games (`_current_season_stats` returns empty), so BOTH teams fall back to `_fallback_stats(2026)` = each team's final 2025 rolling-stat row from `team_stats.parquet`.
3. `get_current_elo_ratings(2026)`: replays cached 2016–2025 schedule + live 2026 schedule through `compute_elo_ratings()`. SEA and NE have had no team-code change in this window, so their Elo history is continuous and correct — **this specific matchup is not affected by the bug in §3.1 below**, only Raiders/Chargers games are.
4. `_build_features()` builds the 32-column feature vector (18 stat diffs + `home_field_context_diff`, `rest_diff`, `injury_impact_diff`, Elo diffs, `market_spread`, `wind_speed`, `short_week_diff`, `away_travel_penalty`, `div_game`, blowout/lookahead diffs).
5. Scored through the saved ensemble model → `home_win_prob`. Spread-calibration regression maps that probability to `implied_spread`. `edge = implied_spread - spread_line`.
6. Confidence-caveat layer checks both teams' `team_roster_turnover()` + `team_fallback_status()` (both true-fallback this week, since it's the season opener) and `team_change_confidence_flag()` (no coach change for either team this offseason) — produces the toss-up-tier nudge only if turnover also clears 35%.
7. `report/cards.py: game_pick_card_html()` renders the final card: win-probability split, edge, spread comparison, any caveat banner.

Every number in that chain is traceable to a raw source above — the one place a viewer sees a number with NO raw-data lineage is the injury-impact weights in §3 (fully hardcoded, never validated against outcomes).

---

## 5. Known, already-documented (not new) design decisions worth flagging for this audit

These aren't bugs — they're deliberate, already-commented tradeoffs already living in the code — listed here so Phase 1–2 doesn't waste time re-litigating them:

- `spread_line`'s open-vs-close ambiguity (nflverse doesn't document which point in the week it reflects) — already investigated against nflverse's own data dictionary, confirmed genuinely unresolvable with current data sources (`model/train.py` comment, ~line 90).
- QB-specific EPA/CPOE, special-teams EPA, pass-rush pressure rate, kickoff temperature — all four were built, ablation-tested, and reverted from `FEATURE_COLS` on real negative walk-forward results (documented inline in `model/train.py`). Still computed, still available for narrative use, deliberately not fed to the model.
- Time-decay sample weighting (recency-weighted training) — built, ablation-tested across 5 half-life settings, underperformed no-decay on every fold. Not implemented.
- `games.dropna(subset=FEATURE_COLS)` at training time — matches `_build_features()`'s live "no spread_line, can't score" rule exactly. No train/serve skew.

---

## 6. What Phase 1–2 of this plan should actually target (the real findings)

### 3.1 — CONFIRMED real bug: team-code relocations break cross-season continuity

`nfl_data_py`'s cached schedule/pbp data uses whichever code was **actually in use** in a given season, not a single retroactively-normalized code:

```
2016: OAK, SD, LA   (Raiders in Oakland, Chargers in San Diego, Rams just moved to LA)
2017–2019: OAK, LAC, LA
2020–2025: LV, LAC, LA
```

Three franchises carry two different code strings within the current 10-season cache (`STL`→`LA` happened in 2016, one season before the cache window starts, so it's not live today but would matter if `HISTORICAL_SEASONS` is ever extended back further).

This silently breaks three things, all confirmed by reading the actual grouping logic (not guessed):

1. **`model/elo.py: compute_elo_ratings()`** — ratings are stored in a plain dict keyed by the raw team-code string with no reset between different codes. When the Raiders' code changed OAK→LV in 2020, `off_ratings.get("LV", INITIAL_RATING)` returned the default 1500 on first use — the ENTIRE 2016–2019 accumulated Elo history for that franchise was silently discarded and replaced with a fresh-expansion-team value, for both the Raiders (2020) and Chargers (2017, one season of history lost). This corrupts the Elo features baked into the **training data** for every Raiders/Chargers game for several seasons after each relocation, and current 2026 ratings for those two teams have had 6 and 9 seasons respectively to rebuild since, diluting but not eliminating the historical distortion in what the model learned to expect from Elo generally.
2. **`data/team_history.py: team_last_n_meetings()`** — filters `schedules["home_team"] == team_a`, so a Raiders/Chiefs "last 5 meetings" query run today would silently miss any meeting that happened while the Raiders were coded "OAK," undercounting real head-to-head history (narrative-only impact, but still wrong).
3. **`data/team_stats.py: build_rolling_team_stats()`**, `ats_win_pct_last5` specifically — this is the one rolling stat deliberately computed by `groupby("team")` alone (not `team, season`), so it correctly bridges season boundaries for a stable-code team, but resets to a cold start for the first few games right after a relocation, since the old code's rows are invisible to the new code's group.

`data/team_stats.py`'s other rolling stats are grouped by `(team, season)` and reset every season regardless, so they're not affected. `data/positional_matchups.py`/`player_trends.py` key off `player_id`, not team code, so they're not affected either.

**Not yet fixed.** The fix (a shared `normalize_team_codes()`, applied at `data/fetch_historical.py`'s cache-build step so the correction lives in the cached parquet files that everything downstream reads) is scoped and ready to build next — see the Phase 1.1 section of this plan.

### 3.2 — Spread sign convention: VERIFIED, not a bug, but undocumented against this plan's own assumption

This plan's stated convention ("negative spread = home favorite," i.e. how a spread reads on a betting ticket) is **not** what this codebase uses, and that's fine — but it needs to be written down explicitly so nobody "fixes" it into a real bug later.

Verified empirically against 15 real 2025 games, cross-checked against `home_moneyline`/`away_moneyline` (unambiguous, since a negative moneyline always means favored, no sign-convention ambiguity possible there): every single row confirms `nfl_data_py`'s raw `spread_line` column is **positive when the home team is favored** (magnitude = home team's expected margin of victory), and every place this codebase touches it (`model/train.py`'s `market_spread = spread_line`, `model/predict.py`'s `implied_spread`, `report/cards.py`'s display) is internally consistent with that same convention. No sign flip is happening anywhere, and none is needed.

This is the opposite of a bettor's-ticket convention (where the favorite's own line carries the minus sign) but it IS internally consistent throughout, and it matches the raw source data without any transformation bug. **A hand-checked regression test locking this in is still worth adding** (Phase 1.2's own ask) — not because a bug was found, but because an undocumented convention is exactly the kind of thing a future change could silently invert.

### 3.3 — `fillna`/`dropna` audit: clean

All 7 `fillna` calls in the training/live-scoring path (`model/train.py`, `model/td_model.py`, `data/situational.py`) already carry an inline justification and are all legitimate "absence means zero effect" cases (no injury listed = 0 impact, dome/no-wind-data = 0 wind, no blowout/lookahead flag = 0), not a single "missing = average" anti-pattern. The 3 `dropna` calls are similarly benign (`FEATURE_COLS` completeness gate, matched between train and serve; a coach/team dedup step). **No fix needed here** — Phase 1.3's paste-prompt can be run as a formality/regression-lock, but it should not find anything new.

### 3.4 — Duplicate-row assertions after merges: not yet added

19 `.merge()` calls across `model/` and `data/`, none currently asserting the output row count or natural-key uniqueness. `model/train.py: build_feature_frame()`'s two team-stat inner-joins (home, then away) are the highest-value place to add this first — a many-to-many blowup there would silently double-count/double-weight training rows. Not yet done.

---

## Bottom line for Phase 1

Two of the four Phase 1 sub-items are effectively already clean (1.3 NaN fills, and 1.2's convention turns out to be correct, just undocumented). The real, confirmed, unfixed bug is 1.1 (team-code relocations corrupting Elo/h2h/rolling continuity) — concrete, scoped, and high-value to fix next. 1.4 (duplicate-row assertions) is pure defensive hardening, not a confirmed active bug, and can be added quickly once 1.1's fix is in.
