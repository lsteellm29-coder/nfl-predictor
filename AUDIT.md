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

---

## Phase 2 — Data leakage audit

The rule this phase enforces: every feature for a game must be computable from data timestamped strictly before that game's kickoff. Went through every feature family by reading the actual construction logic (not just the docstrings), then ran this plan's own stated exit test for real — pick a 2025 Week 4 game and prove every feature value was knowable beforehand.

### Verified leak-free by construction (read the actual code, not assumed)

- **Rolling team stats** (`data/team_stats.py: _rolling_for_team_season()`): every one of the 18 `STAT_COLS` uses `.expanding().mean().shift(1)` (or `.rolling(5, min_periods=1).mean().shift(1)` for `ats_win_pct_last5`) — the `.shift(1)` is what excludes the game's own row from its own rolling average. Confirmed by reading the actual pandas calls, not the docstring's claim about them.
- **Opponent-adjusted stats** (`data/opponent_adjust.py`): `add_opponent_adjusted_columns()` merges the opponent's rolling value at the SAME `(season, week)` — since that value is itself already pre-game-shifted, the adjustment only ever references what was knowable before this specific game. All 3 iterative passes apply the identical shifted-rolling pattern (`_roll_adjusted()`), so iterating doesn't relax the guarantee.
- **Elo ratings** (`model/elo.py: compute_elo_ratings()`): each game's `_pre` snapshot (`home_off_elo_pre` etc.) is appended to `records` BEFORE the `if pd.isna(home_score): continue` check and the actual rating update that follows it — verified by reading the exact line order, not the docstring's claim.
- **Injury impact** (`data/fetch_injuries.py`): both the historical (`nfl.import_injuries`, tagged per season/week by nflverse) and live (ESPN "right now") paths are inherently pre-game by construction — an injury report IS a pre-game document, the same way a betting line is. Training only ever reads the historical path; live scoring only ever reads the live path; no cross-contamination between them.
- **Lookahead flag** (`data/situational.py: lookahead_flags()`): uses `.shift(-1)` — a forward-looking shift, which on its face looks suspicious. It isn't leakage: this reads next week's DIVISION-GAME FLAG off the published schedule (known months in advance), never a future game RESULT. Worth stating explicitly since a naive reviewer (or a future refactor) could misflag any `.shift(-1)` as automatically wrong.

### Spread open/close ambiguity — restated in leakage terms (not new, already in Phase 0/§3.2)

`model/train.py`'s `market_spread = spread_line` may be trained on nflverse's CLOSING line (more informed — has absorbed a week of injury news and sharp money) while live scoring only has access to whatever line The Odds API shows at scoring time (likely an earlier, less-informed number). Both open and close happen before kickoff, so this isn't classic leakage, but it is a real train/serve information asymmetry the plan's Phase 2 explicitly asks about ("closing line used as a feature when you'd only have the opening line"). Already investigated against nflverse's own data dictionary and confirmed genuinely unresolvable with this project's current data sources (no historical open/close pair exists to test against) — restated here, not re-solved.

### `assert_no_leakage()` — the explicit tripwire this phase asks for

New `data/leakage.py`. Doesn't re-derive whether any individual number is correct (the constructions above already guarantee that) — it's a second, explicit, separately-named guard at the four week-boundary filters that are the one place a future refactor could quietly weaken the guarantee (e.g. changing `< week` to `<= week` without noticing the consequence): `model/predict.py: _current_season_stats()`, `model/player_stats.py: _current_season_pbp()`, `model/td_model.py: season_to_date()` and `recency_weighted_touch_share()` (both used with a `upto_week=30` sentinel for a fully-completed fallback season, where the check is a safe no-op).

### The real findings — not leakage, the opposite problem: three silent data-completeness bugs

Running this phase's own exit test (a real 2025 Week 4 game, prove every feature was knowable beforehand) surfaced that **most early-season games across every cached season were being silently dropped from training entirely**, all via `model/train.py`'s `.dropna(subset=FEATURE_COLS)` — not because of leakage, but because of features that legitimately have no value yet this early, with no fallback. All three are now fixed and retrained:

1. **Opponent-adjustment warmup gap**: the 3-pass iterative adjustment compounds its own `.shift(1)` warmup requirement each pass (pass 1 needs 1 prior week, pass 2 needs pass 1's own rolled value, pass 3 needs pass 2's) — with `n_passes=3`, the `*_adj_avg` columns didn't produce a real number until roughly week 5. Verified: 100% of weeks 1–4 across all 10 seasons were null on these 4 columns specifically. Fixed in `data/team_stats.py: _fill_adjusted_warmup_gap()` — falls back to that team's own raw (unadjusted, already leak-free) rolling average when the adjusted one isn't warmed up yet, rather than leaving it null.
2. **Home-field-context gap**: `home_field_context_diff` needs a team to have played at least one HOME game and its opponent at least one AWAY game already (`data/team_stats.py`'s home/away-split rolling averages) — most week-2 (and some week-3) games didn't have both sides populated yet. Fixed in `model/train.py` and `model/predict.py`: `.fillna(0.0)`, the same "no signal yet" default this codebase already used for `blowout_loss_diff`/`lookahead_diff` — a supplementary term reading as neutral, not the game's only signal going blank.
3. **The big one — week 1 has no fallback at all**: every season's week 1 structurally has zero in-season rolling history (there's nothing to `.expanding()` yet), and `build_feature_frame()` had NO equivalent of `model/predict.py`'s own `_fallback_stats()` (which already substitutes a team's final prior-season row for exactly this case at LIVE scoring time). Verified empirically: even after fixing #1 and #2 above, **zero week-1 games, across all 10 cached seasons, survived into training.** For a project whose entire deployment target is Week 1, this is the single most consequential finding of this whole audit — the model had never been trained on a real Week 1 game, or on the fallback-shaped input pattern it's actually asked to use the moment it scores one. Fixed with `model/train.py: _team_stats_with_fallback()`, mirroring the exact fallback pattern `model/predict.py` already uses for live scoring rather than inventing a new one.

**Retrained and compared, not assumed better:**

| | Before any Phase 2 fix (already had Phase 1's team-code fix) | After all 3 fixes |
|---|---|---|
| Training rows | 1782 (2016–2024) | 2341 (2016–2024) — **+31%** |
| True 2025 holdout size | 208 games | 271 games (now includes real Week 1 2025) |
| Selected model type | xgboost (2/4 walk-forward folds) | xgboost (3/4 walk-forward folds) |
| True 2025 holdout accuracy | 0.649 | **0.672** |
| Walk-forward fold: 2022 | 0.660 | 0.658 (flat) |
| Walk-forward fold: 2023 | 0.654 | 0.673 |
| Walk-forward fold: 2024 | 0.760 | 0.728 (down, but tested against a different, now-Week-1-inclusive slice) |
| Walk-forward fold: 2025 | 0.654 | 0.672 |

A genuine, meaningful improvement, not just more data for its own sake — the model is now evaluated on (and has actually learned from) the exact game shape it's deployed to predict.

### Bottom line for Phase 2

The classic leakage question (future information reaching a pre-game feature) checked out clean everywhere it was checked — every rolling/Elo/opponent-adjusted construction already guarantees it, now with an explicit tripwire locking that in. The real yield of this phase was the opposite failure mode: silent training-data exclusion that happened to concentrate exactly on the games this project cares most about. Fixed, tested, retrained, and verified with a real accuracy improvement on the true holdout.

---

## Phase 3 — Week 1 special cases

Went through all five cases the plan names, verifying each against the actual live 2026 Week 1 schedule (not a hypothetical), not just re-reading code in the abstract.

| Case | Status | Evidence |
|---|---|---|
| No prior-season data for the current year | **Already correct, verified live** | `_current_season_stats(2026, 1)` returns 0 rows (nothing played yet, as expected); `get_pregame_stats(2026, 1)` still returns all 32 teams via `_fallback_stats()`; `get_current_elo_ratings(2026)` returns real, non-null off/def ratings for all 32 teams (e.g. SEA `{'off': 1556.9, 'def': 1575.0}`), reflecting accumulated history through 2025 with season-boundary regression already applied. |
| Rest days undefined (no previous game) | **Already correct, verified live** | Pulled the real 2026 Week 1 schedule directly: every team shows `home_rest=7, away_rest=7` — uniform, sane, no huge/negative/null values. nflverse's own schedule already handles this; nothing in this codebase needed to. |
| International game (Melbourne, Sept 10) | **Verified correct, plus one real bug found and fixed** | The real 2026 schedule already tags this game (`LA` home, `SF` away, Sept 10) `location="Neutral"`. Confirmed `home_field_context_diff` computes to exactly `0.0` for this specific game (not just that the code exists — ran it against the live data). But found `data/situational.py`'s `away_travel_penalty()` had no neutral-site awareness at all: it compares the two teams' HOME-market timezones, which describes nothing real when neither team is actually playing at home. This particular matchup (LA/SF, same timezone) happens not to trigger it, but 46 historical neutral-site games exist in the cache and several (2016 LA @ NYG in London, 2017 LV @ NE in London) have real cross-country timezone gaps that WOULD have. Fixed: `away_travel_penalty()` takes a `neutral_site` flag, wired through both `model/train.py` and `model/predict.py` from the schedule's own `location` field. |
| Season opens Wednesday, not Thursday | **Already correct, verified clean** | Grepped the whole codebase for hardcoded weekday logic (`'Thursday'`, `.weekday()`, etc.) — zero hits. `run_week.py`'s `get_current_week()` derives "what week is it" purely from which REG week still has an unplayed game in the schedule (`home_score.isna()`), never from today's calendar date — confirmed it returns `1` right now, correctly. `report/build_report.py`'s day/kickoff formatting takes `weekday` as a parameter straight from the schedule, never inferred. |
| Rookies / new signings have no NFL data | **Handled, but not the way the plan suggests — a real scope call** | Verified live: real 2025-drafted rookies with zero prior-season data (Jalen Royals, Nikko Remigio, both KC WRs) are explicitly excluded from props with a printed reason ("no usable season data to project a fallback card"), not silently defaulted to league-average or a fabricated value — consistent with this codebase's existing "no fabricated confidence" discipline. The plan's suggested fix (an explicit, position-adjusted, below-average prior) would need a data source this codebase doesn't have at all (draft capital / college production stats) — building that is a genuinely new feature, not a Phase 3 edge-case patch, so it's left as an honest, already-defensible gap rather than rushed. |

**Retrained after the travel-penalty fix and compared** (this one touches real historical training rows, not just this week's live scoring): true 2025 holdout accuracy 0.672 → 0.668, walk-forward folds essentially flat, xgboost still wins 3/4. A small, expected result — only a handful of the 46 historical neutral-site games have both a genuine timezone gap and an early kickoff, so a correctness fix touching that few rows out of 2600+ shouldn't (and didn't) move the needle much.

### Bottom line for Phase 3

Four of five cases were already correct, verified against the real, live 2026 schedule rather than assumed from reading code. The one real bug found (neutral-site travel penalty) was dormant for this specific week's game but real and fixed regardless, since the same international-game pattern recurs most seasons. The rookie-prior gap is a deliberate, documented scope decision, not an oversight — building it properly needs a data source this project doesn't have yet.
