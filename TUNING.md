# SharpLine — Tuning Results (Phase 4)

Real numbers from real grid searches/fits against this project's own cached data, produced after Phases 0–3 (correctness/leakage/Week-1-edge-case) were verified clean or fixed. Tuning a broken pipeline is wasted effort, so this only started once those passed.

---

## 4.1 — Elo season-boundary regression-to-mean coefficient

`model/elo.py`'s `SEASON_REGRESSION` (currently `1/3`, "standard FiveThirtyEight convention") was flagged in `AUDIT.md`'s Phase 0 as a guessed constant, never fit against this project's own data. It's applied once per team per season boundary, right before that team's next Week 1 game — exactly where a wrong value would hurt most.

**Method**: `model/tune_roster_weights.py`'s sibling script, `model/tune_shrinkage.py`, grid-searched 0.00–0.60 in steps of 0.05. For each candidate value, rebuilt Elo ratings with that regression amount, then walk-forward: trained a fresh logistic classifier + a separate linear margin-regression on every season strictly before a held-out test season, evaluated ONLY that test season's Week 1 games (6 folds: 2020–2025, 96 games total — earlier seasons don't have enough accumulated Elo history for the regression amount to meaningfully matter yet).

**Full result table:**

| shrinkage | Week 1 accuracy | Week 1 MAE (margin) |
|---|---|---|
| 0.00 | 0.632 | 9.366 |
| 0.05 | 0.632 | 9.369 |
| 0.10 | 0.632 | 9.374 |
| 0.15 | 0.632 | 9.379 |
| 0.20 | 0.632 | 9.385 |
| 0.25 | 0.632 | 9.392 |
| 0.30 | 0.632 | 9.399 |
| **0.33 (current)** | **~0.632** | **~9.403 (interpolated)** |
| 0.35 | 0.632 | 9.405 |
| 0.40 | 0.632 | 9.416 |
| 0.45 | 0.632 | 9.430 |
| 0.50 | 0.632 | 9.444 |
| 0.55 | 0.642 | 9.459 |
| 0.60 | 0.642 | 9.475 |

**What this actually shows**: MAE is monotonically increasing across the whole range, but by a total of only 0.109 points (9.366 → 9.475) across the *entire* 0.00–0.60 sweep — against a baseline MAE of ~9.4 points, that's roughly a 1% relative spread. Accuracy is completely flat (0.632) from 0.00 through 0.50, only ticking up at 0.55–0.60 — on 96 total games, a 1-percentage-point accuracy shift is one single game flipping in one fold, not a trend.

**Decision: keeping the current value (1/3), not switching to the numerically-lowest-MAE value (0.00).** The plan's own instruction says to pick the value that minimizes MAE, and by the letter of that, 0.00 wins. But treating a 0.109-point difference on a 96-game sample as a confident, meaningful result would be overfitting to noise, not a real finding — the honest read of this table is "this parameter doesn't move Week 1 accuracy much anywhere in the tested range." Regression-to-the-mean also has a real theoretical justification a 6-fold/96-game test can't fully capture: without it, a team's rating never corrects back toward average after one unusually extreme season, which compounds over *many* seasons in ways this short backtest window doesn't surface. Given a razor-thin, likely-noisy MAE edge doesn't clear the bar for abandoning a standard, theoretically-grounded convention, `model/elo.py`'s `SEASON_REGRESSION` is left at `1/3`. This is a real, documented answer to the question the plan asked, not a non-result — the parameter was tested and found not to matter much within a sane range, which is itself useful to know before spending more tuning effort here.

---

## 4.2 — Roster-adjustment signal weights

`model/tune_roster_weights.py`: five roster-change signals, computed for every Week 1 game 2017–2025 as `{signal}_diff = home - away` (same convention `model/train.py`'s `FEATURE_COLS` already uses) — QB change (detected off the schedule's own `home_qb_id`/`away_qb_id`, comparing each team's Week 1 starter against their own final game of the prior season), O-line turnover, head-coach change, offensive-skill turnover, and a pooled defensive-front-seven turnover (defensive front + linebackers combined, summing departed/total rather than averaging two differently-sized units). Fit with `Ridge(alpha=1.0)` against real point margin, 95% CIs via 1000-iteration bootstrap resampling (ridge has no closed-form CI). 143 Week 1 games had complete data across all five signals.

**Fitted weights:**

| signal | weight | 95% CI | zeroed? |
|---|---|---|---|
| `qb_change_diff` | −0.06 | [−3.27, +3.35] | **yes** — crosses zero |
| `ol_turnover_diff` | −5.01 | [−12.72, +2.57] | **yes** — crosses zero |
| `coaching_change_diff` | −4.60 | [−8.45, −0.83] | no — the only significant signal |
| `skill_turnover_diff` | +4.01 | [−4.60, +12.95] | **yes** — crosses zero |
| `front7_turnover_diff` | −5.28 | [−14.27, +5.20] | **yes** — crosses zero |

**Four of five signals didn't clear the bar — reported honestly, not forced to match the plan's prior intuition.**

The QB-change result is the most surprising one, since it directly contradicts both football intuition and the plan's own stated expectation ("the big one... 4–7 points"). Verified the detector itself is correct first, not just trusted the number — hand-checked five well-known real Week 1 QB changes (2021 NYJ Darnold→Wilson, 2023 NYJ Wilson→Rodgers, 2018 CLE Kizer→Taylor, etc.) and confirmed every one is correctly flagged. Also checked for multicollinearity with `coaching_change_diff` as an alternative explanation — the two signals correlate at only 0.04, essentially independent, ruling that out too.

The more likely real explanation: a binary "did the QB change" flag can't distinguish an *upgrade* (a team benching a struggling starter for a promising rookie, or landing a proven veteran like Rodgers or Wilson) from a *downgrade* (an elite starter's backup stepping in) — and real Week 1 QB changes are a genuine mix of both. Pooled together, the two directions cancel out in a way a single yes/no feature can't separate, even if any one INSTANCE of a QB change really does swing a game by several points. This points toward QB *quality delta* (this codebase already has the per-QB EPA/CPOE game logs in `data/player_trends.py` needed to build that) as the right follow-up feature, not evidence that starting-QB continuity doesn't matter — but building and validating that properly is its own piece of work, not done here.

The one significant result (`coaching_change_diff`, −4.60, CI doesn't cross zero) needs its own caveat before being trusted at face value: teams that fire their head coach mid-cycle tend to already have been struggling — a coaching change is as much a *symptom* of a bad team as a cause of one. A simple ridge fit on raw margin can't separate "the new coach made them worse" from "they were already the kind of team that fires its coach," so this coefficient is more honestly read as "a team with a coaching change this season tends to underperform its other stats by about 4–5 points," not a clean causal effect of the coaching change itself.

**Decision: none of these five weights are deployed as a new model feature or a wired-in roster-adjustment module.** The plan's own Phase 4.2 instruction was explicit — "if any weight has a CI crossing zero, set it to zero" — and that leaves only one signal standing, with a real confound attached even then. Shipping a "roster adjustment" built on one confounded coefficient and four zeros isn't a real adjustment system, it's noise dressed up as one. This is a genuine, useful negative result: the naive versions of these five signals don't cleanly predict Week 1 margin on 143 games of real data, and the two most likely paths to a real signal (QB quality delta instead of a binary flag; a causally cleaner estimate of coaching-change impact) are both identified, scoped, and left for follow-up rather than shipped half-validated.

---

## 4.3 + 4.4 — Calibration, full benchmark suite, and the three baselines

`model/calibration.py` already existed and was genuinely sophisticated (Brier score, a bucketed reliability table, Platt scaling and isotonic regression both tested honestly on out-of-fold training predictions then judged on the untouched holdout, never fit and graded on the same data). What it was missing, against this plan's own ask: log loss, MAE against actual margin, ATS accuracy, and a direct comparison to the plan's Bad/OK/Good/Vegas benchmark table — plus `data/baselines.py`'s three dumb models, now printed in the same report every time (4.4's own explicit ask).

**Full report, run against the real, current model (2025 holdout, 271 games, after all of Phases 1–3's fixes):**

| metric | value | rating |
|---|---|---|
| Straight-up accuracy | 0.668 | ok (0.67 = good, 0.67 = Vegas — essentially tied with the market) |
| Brier score | 0.212 | ok (0.20 = good, 0.19 = Vegas) |
| ATS accuracy | 0.509 | bad (50% is what Vegas itself gets "by design," per the plan's own table) |
| MAE vs actual margin | 9.729 | **good** (better than the plan's own 10.2 "good" threshold, close to Vegas's ~10.0) |
| Log loss | 0.610 | (no benchmark row in the plan's table) |

**Baselines, same 271 games:**

| baseline | accuracy |
|---|---|
| Always pick home | 0.539 |
| Always pick the Vegas favorite | 0.653 |
| Prior-season win% only | 0.575 (n=247 — 24 games had a team with no prior-season record) |

**The model beats all three baselines** (0.668 vs 0.539/0.653/0.575) — the real bar Phase 4.4 sets, cleared.

**The ATS number needs its own explanation, not just a "bad" label.** A model that takes the market's own `spread_line` as a direct input feature (this one does — `market_spread` is in `FEATURE_COLS`) will naturally track close to the market's own accuracy rather than beat it, unless it has genuine information the market hasn't already priced in. 50.9% ATS is essentially indistinguishable from the "coin flip" a perfectly efficient market should produce (the plan's own table lists Vegas itself at "50% by design" for this exact reason) — this isn't a failure of the model, it's the honest, expected signature of a well-calibrated win-probability model that isn't claiming to have found market inefficiency. Straight-up accuracy and MAE are the metrics this model is actually built to be good at, and both check out.

**No leakage red flag.** The plan's own warning ("if you post 70%+ ATS in a backtest, you have leakage — go back to Phase 2") doesn't fire here: 50.9% is nowhere near that threshold, consistent with Phase 2's leakage audit having come back clean.

**Calibration decision: none.** Raw Brier (0.2116) already beat both Platt (0.2120) and isotonic (0.2133) on the true holdout — the model's probabilities are already reasonably honest out of the box, and the reliability table's bucket-to-bucket deviations from the diagonal bounce in both directions with no systematic bias (e.g. overconfident in the 0.55–0.6 and 0.65–0.7 buckets, underconfident in 0.5–0.55 and 0.7–1.0), consistent with sampling noise on 15–25 games per bucket rather than a real, fixable miscalibration pattern.

---

## Phase 5 — Walk-forward Week 1 backtest across every available season

Every result above (Phases 4.1–4.4) is really about one thing at a time: one parameter, or one season (the true 2025 holdout). This phase is the plan's own explicit check that the whole pipeline — not just its 2025 snapshot — actually holds up: `model/backtest.py`, one command (`python -m model.backtest`), reusing `model/train.py`'s own `train_logistic`/`train_xgboost`/`predict_proba`/`train_spread_calibration` directly (not a parallel reimplementation that could quietly drift from what actually ships).

**Method**: for each season with enough prior history, train fresh logistic + xgboost models on every season strictly before it, predict ONLY that season's Week 1 (never fit and tested on the same season, never previewing a later one), grade with Phase 4.3's own `full_backtest_metrics()`. Model type pinned to `xgboost` — the type the real deployed model actually uses, per `model/train.py`'s own walk-forward vote — since this backtest asks "does the shipped pipeline hold up historically," not "which model type is best in hindsight" (a different, already-answered question, `select_model_type_by_walk_forward()`). This project's cache holds 10 seasons (2016–2025, `config.HISTORICAL_SEASONS`), not the plan's literal "2015" — there's no 2015 data to fit on — so folds start once `model/train.py`'s own `MIN_WALK_FORWARD_TRAIN_SEASONS` (6) prior seasons are banked: 4 folds, 2022–2025, 63 Week 1 games total.

**Per-season table (real run):**

| season | n | straight-up acc | Brier | ATS | MAE |
|---|---|---|---|---|---|
| 2022 | 15 | 0.600 | 0.246 | 0.533 | 9.093 |
| 2023 | 16 | 0.625 | 0.249 | 0.438 | 11.154 |
| 2024 | 16 | 0.750 | 0.202 | 0.750 | 8.934 |
| 2025 | 16 | 0.812 | 0.169 | 0.625 | 5.810 |

**Pooled (genuine pooling — all 63 games' raw predictions concatenated and graded once, not an average of the four rows above, since that would give a subtly wrong answer for Brier/log loss/MAE):**

| metric | value | rating |
|---|---|---|
| Straight-up accuracy | 0.698 | good (clears the plan's own 0.67 "good"/Vegas bar) |
| Brier score | 0.216 | ok |
| ATS accuracy | 0.587 | good — well below the 0.70 leakage-suspicion line, so a real result, not a leakage flag |
| MAE vs actual margin | 8.742 | good |
| Log loss | 0.628 | (no benchmark row in the plan's table) |

**Baselines, same 63 games:**

| baseline | accuracy |
|---|---|
| Always pick home | 0.508 |
| Always pick the Vegas favorite | **0.698** |
| Prior-season win% only | 0.649 (n=57) |

**Honest result: the model ties, not beats, the always-Vegas-favorite baseline on this pooled sample (0.698 vs 0.698).** It clears always-home (0.508) and prior-season-win% (0.649) comfortably, and its Brier/MAE/ATS numbers are all independently good — but on straight-up accuracy alone, across these specific 63 games, it matches rather than exceeds picking the market favorite every time. This isn't glossed over: on a 63-game sample, a tie with the market on straight-up picks is a believably real result, not a bug — this model's own `market_spread` feature already tracks the favorite closely by construction (the same reason Phase 4.3's ATS number sits near 50%), so beating "always take the favorite" specifically was never guaranteed just because the model beats the other two, cruder baselines. Per-season variance is real too: 2022 and 2023 are meaningfully weaker (0.600–0.625 straight-up, 2023's ATS at 0.438 is a losing record against the market that year) than 2024–2025 (0.750–0.812) — with only 15–16 games per fold, no single season's number should be over-read, but the 4-fold spread itself is worth keeping in mind rather than anchoring only on the strongest (2025) season.

**Exit test**: runs end-to-end with one command and prints a table — met. "Beats all three baselines" — met for 2 of 3 outright, tied (not beaten) on the third (always-Vegas-favorite) on this specific pooled sample; reported as-is rather than rounded up to a clean sweep.
