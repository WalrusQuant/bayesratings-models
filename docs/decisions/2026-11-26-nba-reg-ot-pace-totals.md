# 2026-11-26 — NBA regulation/OT pace totals

Opt-in pace-and-efficiency model for NBA expected points (`BAYESRATINGS_NBA_TOTALS=pace`;
default stays `ols` pending a full multi-season replay). Ratings, win
probabilities, and spreads stay the same.

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
  the SQLite schema does not change. Because `_pending()` checks a single
  global `VERSION`, bumping `VERSION` to 5 re-extracts stored `game_summaries`
  for all four leagues (`mlb`, `nba`, `nfl`, `nhl`) once on the next
  `extract` or `daily` run (local summaries only, no ESPN calls).
- `bayesratings/config.py`: adds `nba_points_model()` backed by
  `BAYESRATINGS_NBA_TOTALS` (`ols` by default, `pace` to opt in).
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
  - If `poss` is missing or fewer than 15 non-OT games have valid box scores,
    `fit_nba_pace_totals` returns `None` and the forecast falls back to the
    OLS blend.
- `bayesratings/hybrid.py`, `forecast.py`, `daily.py`, `replay.py`: adds
  `"pace"` to `VALID_POINTS_MODELS`, routes NBA through `config.nba_points_model()`,
  attaches NBA box-score columns before `run_model` when `pace` is enabled,
  and adds the OT lift after neutral-site margin averaging so swapped neutral
  games stay symmetric and team totals still line up with the spread.
- `scripts/eval_nba_totals.py`: read-only walk-forward evaluator against a
  SQLite database for multi-season comparison (`ols` vs `pace`, weekly or
  daily refit, day-clustered 95% bootstrap CIs by season and pooled).

## Walk-forward results

Weekly refit (train strictly on games dated before Tuesday, 90-day half-life
decay, fast weighted-ridge ratings shared by both totals models). Intervals
are paired 95% cluster-bootstrap CIs by game day (2,000 resamples, seed
20261126). Delta is `pace` minus `ols` on the exact same games (negative is
better). Regular season and postseason, 2023-24 through 2026-11-26 (`4,214`
scored games, `661` days; `4,590` total NBA games with v5 `team_game_stats`
including late 2022-23 warm-up).

```bash
python -m bayesratings.team_counts --db data/bayesratings.db --leagues nba
python scripts/eval_nba_totals.py --db data/bayesratings.db --seasons 2024,2025,2026,2027
```

| Season | Variant | Games | Days | MAE | Δ MAE | 95% CI (Δ MAE) | RMSE | Δ RMSE | Bias |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| 2024 | `ols` | 1313 | 205 | 15.034 | +0.000 | - | 18.870 | +0.000 | +1.940 |
| 2024 | `pace` | 1313 | 205 | 15.059 | +0.025 | [-0.100, +0.148] | 18.835 | -0.035 | +2.020 |
| 2025 | `ols` | 1315 | 211 | 14.879 | +0.000 | - | 18.598 | +0.000 | -0.629 |
| 2025 | `pace` | 1315 | 211 | 14.764 | -0.115 | [-0.216, -0.014] | 18.493 | -0.105 | -0.556 |
| 2026 | `ols` | 1316 | 209 | 15.197 | +0.000 | - | 19.219 | +0.000 | -0.009 |
| 2026 | `pace` | 1316 | 209 | 15.148 | -0.049 | [-0.151, +0.056] | 19.121 | -0.098 | -0.054 |
| 2027 | `ols` | 270 | 36 | 15.559 | +0.000 | - | 19.487 | +0.000 | +2.625 |
| 2027 | `pace` | 270 | 36 | 15.146 | -0.413 | [-0.696, -0.115] | 19.062 | -0.426 | +2.426 |
| all | `ols` | 4214 | 661 | 15.070 | +0.000 | - | 18.936 | +0.000 | +0.573 |
| all | `pace` | 4214 | 661 | 15.001 | -0.070 | [-0.132, -0.007] | 18.834 | -0.102 | +0.595 |

RMSE improves in all four seasons (`-0.102` pooled). Pooled MAE drops by
`-0.070` with a 95% day-cluster CI excluding zero (`[-0.132, -0.007]`), with
the largest gain early in a season (`2027: -0.413` MAE) where unshrunk `OS`
and `DS` are noisiest. Pass `--use-stan` to `scripts/eval_nba_totals.py` to
re-run with Stan Bradley-Terry ratings instead of the ridge stand-in.
`tests/test_points_model.py` and `tests/test_nba_totals.py` lock the model
behavior, opt-in toggle, and invariants.
