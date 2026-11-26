"""Per-league expected-points model: MLB negative binomial, NHL Poisson,
NBA OLS blend by default (opt-in regulation/OT pace-efficiency via
BAYESRATINGS_NBA_TOTALS=pace), NFL keeps the OLS blend. Win probabilities
and spreads are unchanged."""
import os
import numpy as np
import pandas as pd
from bayesratings.games import validate_games
from bayesratings.forecast import forecast_games
from bayesratings.hybrid import fit_hybrid_models, points_model_for, _fit_poisson, _nb_alpha
from bayesratings.team_stats import compute_team_stats

# 1. league mapping (default + NBA opt-in toggle)
assert points_model_for("mlb") == "negbin" and points_model_for("MLB") == "negbin"
assert points_model_for("nhl") == "poisson"
assert points_model_for("nfl") == "ols"
_prev_nba = os.environ.pop("BAYESRATINGS_NBA_TOTALS", None)
try:
    assert points_model_for("nba") == "ols" and points_model_for("NBA") == "ols"
    os.environ["BAYESRATINGS_NBA_TOTALS"] = "pace"
    assert points_model_for("nba") == "pace" and points_model_for("NBA") == "pace"
    os.environ["BAYESRATINGS_NBA_TOTALS"] = "ols"
    assert points_model_for("nba") == "ols"
finally:
    if _prev_nba is None:
        os.environ.pop("BAYESRATINGS_NBA_TOTALS", None)
    else:
        os.environ["BAYESRATINGS_NBA_TOTALS"] = _prev_nba
print("ok league mapping")

# 2. Poisson IRLS recovers known coefficients
rng = np.random.default_rng(1)
X = np.column_stack([np.ones(20000), rng.normal(size=(20000, 5)) * 0.5])
true = np.array([1.2, 0.3, -0.2, 0.1, 0.0, 0.25])
y = rng.poisson(np.exp(X @ true))
assert np.allclose(_fit_poisson(X, y), true, atol=0.03)
assert _nb_alpha(y, np.exp(X @ true)) < 0.02          # Poisson data: ~no dispersion
r = 4.0; mu = np.exp(X @ true)
y_nb = rng.negative_binomial(r, r / (r + mu))
assert abs(_nb_alpha(y_nb, mu) - 1 / r) < 0.04        # NB data: alpha ~ 1/r
print("ok glm fits")

# 3. synthetic low-scoring league
teams = [str(i) for i in range(10)]
s = dict(zip(teams, rng.normal(0, 0.3, 10)))
rows = []
for k in range(1500):
    h, a = rng.choice(teams, 2, replace=False)
    n = int(k % 15 == 0)
    hs = rng.poisson(np.exp(1.4 + s[h] - s[a] + (0 if n else 0.08)))
    as_ = rng.poisson(np.exp(1.4 + s[a] - s[h]))
    if hs == as_: hs += 1
    rows.append({"season": 2026, "date": f"2026-0{1 + k % 6}-{1 + k % 28:02d}", "home_team": h,
                 "away_team": a, "home_score": hs, "away_score": as_, "neutral": n})
games = validate_games(pd.DataFrame(rows))
ratings = pd.Series({t: v - np.mean(list(s.values())) for t, v in s.items()})
stats = compute_team_stats(games)
sched = validate_games(pd.DataFrame([
    {"season": 2026, "date": "2026-07-01", "home_team": "0", "away_team": "1",
     "home_score": np.nan, "away_score": np.nan, "neutral": 0},
    {"season": 2026, "date": "2026-07-01", "home_team": "2", "away_team": "3",
     "home_score": np.nan, "away_score": np.nan, "neutral": 1}]))
sw = sched.rename(columns={"home_team": "away_team", "away_team": "home_team"})
fc = {}
for pm in ("ols", "poisson", "negbin", "pace"):
    hyb = fit_hybrid_models(games, ratings, stats["team_stats"], points_model=pm)
    assert hyb["points_model"] == pm
    fc[pm] = forecast_games(sched, ratings, stats["team_stats"], hyb, stats["league_avg_points"])
    f2 = forecast_games(sw, ratings, stats["team_stats"], hyb, stats["league_avg_points"])
    # neutral game stays symmetric under a home/away swap
    assert abs(fc[pm].home_total_pred[1] - f2.away_total_pred[1]) < 0.11
    assert abs(fc[pm].home_prob[1] - f2.away_prob[1]) < 1e-3
    assert (fc[pm].total_pred - fc[pm].home_total_pred - fc[pm].away_total_pred).abs().max() < 0.11
    if pm in ("poisson", "negbin"):
        assert hyb["home_count_fit"].family == pm
        if pm == "poisson":
            assert hyb["home_count_fit"].alpha == 0 and hyb["away_count_fit"].alpha == 0
    if pm == "pace":
        # score-only synthetic frame lacks box-score possessions -> falls back to OLS
        assert hyb["pace_fit"] is None
    # team points line up with the spread (margin reconciliation)
    diff = fc[pm].home_total_pred - fc[pm].away_total_pred
    assert (diff + fc[pm].home_spread).abs().max() < 0.21, pm
# win probabilities and spreads identical across points models
for pm in ("poisson", "negbin", "pace"):
    for c in ("home_prob", "away_prob", "home_spread", "home_odds"):
        assert (fc[pm][c].values == fc["ols"][c].values).all(), c
# score-only pace fallback matches OLS totals exactly
assert (fc["pace"].total_pred.values == fc["ols"].total_pred.values).all()
# Poisson and NB share the same mean model
assert (fc["poisson"].total_pred.values == fc["negbin"].total_pred.values).all()
# count model totals differ from the OLS blend
assert not (fc["poisson"].total_pred.values == fc["ols"].total_pred.values).all()
print("ok forecasts")

try:
    fit_hybrid_models(games, ratings, stats["team_stats"], points_model="bogus")
    raise AssertionError("bad points_model accepted")
except ValueError:
    pass
print("ok all")
