# 2026-11-26 — NBA regulation/OT pace totals

Walk-forward test of a pace-and-efficiency model for NBA totals
(`POINTS_MODELS["nba"] = "pace"`). Ratings, win probabilities, and spreads
stay the same.

## Why change it

The current OLS + `OS * DS` blend trains on raw final scores and unadjusted
per-game scoring averages. Two things hurt it in NBA:

- Roughly 6% of NBA games go to overtime, adding around 22.5 points and 10.5
  possessions. Inside a 90-day half-life window, a couple of recent OT games
  inflate a team's `OS` and `DS`, and the OLS intercept spreads the average
  OT bump across every game whether the spread is 1 or 15.
- Both teams share the same game possessions. Per-game `OS` and `DS` mix pace
  with shooting efficiency and make no adjustment for who each team played.

## What changed

- `bayesratings/team_counts.py` (`VERSION = 5`): appends `reg_pts` (index 18
  of `COUNTS["nba"]`), the sum of each team's first four quarter linescores
  from the stored ESPN summary (`value`, falling back to `displayValue`).
  Indices `0..17` stay where they are (`poss` at 16, `minutes` at 17), and
  the SQLite schema does not change.
- `bayesratings/nba_totals.py`:
  - `attach_nba_box_cols(conn, games)` joins `poss`, `minutes`,
    `reg_home_score`, and `reg_away_score` from `team_game_stats`. It skips
    `season_type = 0` (the NBA Cup championship) and still reads older v4 rows
    by taking final scores as regulation scores whenever `minutes <= 48.5`.
  - `fit_nba_pace_totals(games, sample_weight)` fits three pieces on the same
    90-day decay window and weights as the rating model:
    1. Weighted Ridge on non-OT games for 48-minute regulation possessions:
       `poss = mu_pace + tau[home] + tau[away]` (`lam_pace = 3.0`).
    2. Possession-weighted Ridge (`w * poss / mu_pace`) on non-OT games for
       regulation offensive and defensive efficiency (points per 100
       possessions):
       `eff_h = mu_eff + hca_eff * (1 - neutral) + off[home] + def[away]`,
       `eff_a = mu_eff - hca_eff * (1 - neutral) + off[away] + def[home]`
       (`lam_eff = 8.0`). Weighting by possessions keeps total weighted
       regulation points conserved.
    3. An overtime lift fit from the training window and shrunk toward a 6%
       OT rate and 22.5 extra points:
       `p_ot(m) = clip(kappa * [Phi((0.5 - m) / sigma_m) - Phi((-0.5 - m) / sigma_m)], 0, 0.25)`,
       multiplied by `mu_ot`.
  - If `poss` is missing or fewer than 15 non-OT games have valid possessions,
    `fit_nba_pace_totals` returns `None` and the forecast falls back to the
    OLS blend.
- `bayesratings/hybrid.py`, `forecast.py`, `daily.py`, `replay.py`: adds
  `"pace"` to `VALID_POINTS_MODELS`, sets `POINTS_MODELS["nba"] = "pace"`,
  attaches NBA box-score columns before `run_model`, and adds the OT lift
  after neutral-site margin averaging so swapped neutral games stay symmetric
  and team totals still line up with the spread.

## Walk-forward results

Each test game is scored using only games dated strictly before it, with the
90-day half-life decay. Intervals are 95% cluster-bootstrap CIs by game day
(2,000 resamples). Delta is `pace` minus `ols` on the exact same games
(negative is better).

### Real NBA 2026-27 (194 games, 27 days)

Both models trained on the exact same completed 2026-27 games with box-score
possessions, refit daily after a 60-game warm-up.

| Variant | MAE | Δ MAE | 95% CI (Δ MAE) | RMSE | Δ RMSE | Bias |
|---|---:|---:|---|---:|---:|---:|
| `ols` baseline | 16.387 | +0.000 | - | 20.487 | +0.000 | +0.747 |
| `pace` (`lam_pace=2.5, lam_eff=7.0`) | 14.815 | -1.572 | [-2.313, -0.908] | 19.049 | -1.438 | +0.454 |
| `pace` (`lam_pace=3.0, lam_eff=8.0`, kept) | 14.759 | -1.627 | [-2.426, -0.922] | 19.025 | -1.462 | +0.436 |

Early-season `ols` is noisy because unshrunk `OS` and `DS` swing hard on a
few games and a few OT finishes, so the gap here is larger than it will be
late in a season.

### 4-season synthetic NBA (4,575 games, 651 days)

30 teams, 65% intra-conference schedule, season-to-season drift, NBA-scale
pace and efficiency noise, ~4.5-6% overtime rate, weekly refit matching
`replay.py`.

| Variant | MAE | Δ MAE | 95% CI (Δ MAE) | RMSE | Δ RMSE |
|---|---:|---:|---|---:|---:|
| `ols` baseline | 14.831 | +0.000 | - | 18.845 | +0.000 |
| `pace` (`lam_pace=2.5, lam_eff=7.0`) | 14.728 | -0.103 | [-0.166, -0.041] | 18.704 | -0.141 |
| `pace` (`lam_pace=3.0, lam_eff=8.0`, kept) | 14.728 | -0.104 | [-0.169, -0.039] | 18.706 | -0.139 |

`lam_pace = 3.0` and `lam_eff = 8.0` sit close to the theoretical
`sigma_noise^2 / sigma_team^2` ratio for the 90-day decay window. Both MAE
and RMSE drop on both tests with 95% intervals excluding zero.
`tests/test_points_model.py` and `tests/test_nba_totals.py` lock the model
behavior and invariants.
