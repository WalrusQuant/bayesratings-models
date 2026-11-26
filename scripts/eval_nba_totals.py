"""Walk-forward comparison of NBA totals: OLS blend vs regulation/OT pace model.

Reads completed NBA games and `team_game_stats` from a SQLite database
(read-only, writes nothing) and scores `ols` against `pace` strictly
out-of-sample on the same 90-day half-life decay window used by the pipeline.

    python scripts/eval_nba_totals.py --db data/bayesratings.db
    python scripts/eval_nba_totals.py --db data/bayesratings.db --seasons 2024,2025,2026,2027
    python scripts/eval_nba_totals.py --db data/bayesratings.db --refit daily --lam-pace 3.0 --lam-eff 8.0

By default, ratings use a fast weighted ridge stand-in so multi-season runs
finish in seconds (both totals models see the exact same ratings and margin
predictions). Pass `--use-stan` to run Stan Bradley-Terry ratings instead.
"""

import argparse
import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from bayesratings import db, nba_totals
from bayesratings.config import db_path
from bayesratings.forecast import forecast_games
from bayesratings.games import validate_games
from bayesratings.hybrid import fit_hybrid_models
from bayesratings.replay import week_anchor
from bayesratings.team_stats import compute_team_stats
from bayesratings.weights import decay_window, half_life_for


def _ridge_ratings(train_df: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Fast weighted ridge team strength stand-in for multi-season sweeps."""
    teams = sorted(set(train_df["home_team"]) | set(train_df["away_team"]))
    ti = {t: i for i, t in enumerate(teams)}
    t_n = len(teams)
    x = np.zeros((len(train_df), t_n + 1), dtype=float)
    neut = (train_df["neutral"].fillna(0).to_numpy(dtype=float)
            if "neutral" in train_df.columns else np.zeros(len(train_df)))
    for k, (h, a, n_flag) in enumerate(zip(train_df["home_team"], train_df["away_team"], neut)):
        x[k, ti[h]] = 1.0
        x[k, ti[a]] = -1.0
        x[k, t_n] = 1.0 - n_flag
    y = (train_df["home_score"] - train_df["away_score"]).to_numpy(dtype=float)
    pen = np.eye(t_n + 1) * 10.0
    pen[t_n, t_n] = 1.0
    beta = np.linalg.solve((x.T * weights) @ x + pen, (x.T * weights) @ y)
    r = beta[:t_n] - np.mean(beta[:t_n])
    return pd.Series(r / 10.0, index=teams)


def _stan_ratings(train_df: pd.DataFrame, stan_model) -> pd.Series:
    from bayesratings.ratings import attach_b2b, fit_bt_ratings
    from bayesratings.weights import league_fit_kwargs

    kw = league_fit_kwargs("nba")
    df = attach_b2b(train_df) if kw["b2b"] else train_df
    fit = fit_bt_ratings(
        df, stan_model, evaluate=False,
        iter=400, warmup=200, chains=2, seed=123,
        b2b=kw["b2b"], sigma_scale=kw["sigma_scale"],
        margin_home=kw["margin_home"], starter=kw["starter"],
    )
    return fit["ratings"]


def _cluster_boot_ci(dates, delta, n_boot=2000, seed=20261126):
    rng = np.random.default_rng(seed)
    groups = [g["x"].to_numpy() for _, g in pd.DataFrame({"d": dates, "x": delta}).groupby("d")]
    n_g = len(groups)
    if n_g == 0:
        return float("nan"), float("nan")
    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n_g, size=n_g)
        means[b] = float(np.mean(np.concatenate([groups[i] for i in idx])))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def load_nba_games(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load completed NBA games with box-score pace/minutes/reg_pts attached."""
    games = db.read_games(conn, "nba", event_id=True)
    done = games.dropna(subset=["home_score", "away_score"]).copy()
    done = nba_totals.attach_nba_box_cols(conn, done)
    done = done.drop(columns=["espn_event_id"], errors="ignore")
    done = validate_games(done).sort_values("date").reset_index(drop=True)
    return done


def evaluate_walkforward(
    df: pd.DataFrame,
    seasons=None,
    refit: str = "weekly",
    lam_pace: float = nba_totals.LAM_PACE,
    lam_eff: float = nba_totals.LAM_EFF,
    min_train: int = 80,
    half_life_days: int = None,
    use_stan: bool = False,
) -> pd.DataFrame:
    """Run strictly out-of-sample walk-forward scoring on `df`.

    Returns a DataFrame with one row per scored game: season, date, actual,
    ols_pred, pace_pred, abs_margin, is_ot.
    """
    hl = half_life_for("nba") if half_life_days is None else int(half_life_days)
    stan_model = None
    if use_stan:
        from bayesratings.ratings import get_bt_model
        stan_model = get_bt_model()

    season_set = set(seasons) if seasons else None
    df = df.copy()
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")
    all_dates = sorted(df["date_str"].unique())
    if season_set is not None:
        eval_dates = sorted(df.loc[df["season"].isin(season_set), "date_str"].unique())
    else:
        eval_dates = all_dates

    # Group test dates by fit cutoff (weekly Tuesday anchor or daily).
    buckets = {}
    for d in eval_dates:
        cut = week_anchor(d) if refit == "weekly" else d
        buckets.setdefault(cut, []).append(d)

    out_rows = []
    for cut in sorted(buckets):
        train_all = df[df["date_str"] < cut]
        if len(train_all) < min_train:
            continue
        has_poss = ("poss" in train_all.columns
                    and int(train_all["poss"].notna().sum()) >= min_train)
        if not has_poss:
            continue

        win = decay_window(train_all, half_life_days=hl)
        w = win["time_weight"].to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ratings = (_stan_ratings(win, stan_model) if use_stan
                       else _ridge_ratings(win, w))
            stats = compute_team_stats(win, sample_weight=w)
            hyb_ols = fit_hybrid_models(
                win, ratings, stats["team_stats"], points_model="ols", sample_weight=w
            )
            pace_fit = nba_totals.fit_nba_pace_totals(
                win[win["neutral"].fillna(0).astype(int) == 0],
                sample_weight=w[win["neutral"].fillna(0).astype(int).to_numpy() == 0],
                lam_pace=lam_pace,
                lam_eff=lam_eff,
            )
            if pace_fit is None:
                continue
            hyb_pace = dict(hyb_ols, points_model="pace", pace_fit=pace_fit)

        test = df[df["date_str"].isin(buckets[cut])]
        if season_set is not None:
            test = test[test["season"].isin(season_set)]
        if len(test) == 0:
            continue

        sched = test[["season", "date", "home_team", "away_team", "neutral"]].copy()
        sched["home_score"] = np.nan
        sched["away_score"] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc_ols = forecast_games(
                sched, ratings, stats["team_stats"], hyb_ols, stats["league_avg_points"]
            )
            fc_pace = forecast_games(
                sched, ratings, stats["team_stats"], hyb_pace, stats["league_avg_points"]
            )

        mins = (test["minutes"].to_numpy(dtype=float)
                if "minutes" in test.columns else np.full(len(test), np.nan))
        actual = (test["home_score"] + test["away_score"]).to_numpy(dtype=float)
        ols_p = fc_ols["total_pred"].to_numpy(dtype=float)
        pace_p = fc_pace["total_pred"].to_numpy(dtype=float)
        abs_m = np.abs(fc_ols["home_spread"].to_numpy(dtype=float))
        is_ot = np.where(np.isfinite(mins), mins > 48.5, False)

        for i in range(len(test)):
            out_rows.append({
                "season": int(test["season"].iloc[i]),
                "date": test["date_str"].iloc[i],
                "actual": actual[i],
                "ols_pred": ols_p[i],
                "pace_pred": pace_p[i],
                "abs_margin": abs_m[i],
                "is_ot": bool(is_ot[i]),
            })

    return pd.DataFrame(out_rows)


def format_report(scored: pd.DataFrame, n_boot: int = 2000, seed: int = 20261126) -> str:
    if len(scored) == 0:
        return "No games scored (check DB path, seasons, and team_game_stats coverage)."
    lines = [
        "| Season | Variant | Games | Days | MAE | Δ MAE | 95% CI (Δ MAE) | RMSE | Δ RMSE | Bias | Close Bias | Blowout Bias |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    slices = [(str(s), scored[scored["season"] == s]) for s in sorted(scored["season"].unique())]
    if len(slices) > 1:
        slices.append(("all", scored))

    for label, g in slices:
        y = g["actual"].to_numpy()
        d = g["date"].to_numpy()
        sp = g["abs_margin"].to_numpy()
        n_games = len(g)
        n_days = len(set(d))
        p_ols = g["ols_pred"].to_numpy()
        p_pace = g["pace_pred"].to_numpy()
        ae_ols = np.abs(p_ols - y)
        ae_pace = np.abs(p_pace - y)
        rmse_ols = float(np.sqrt(np.mean((p_ols - y) ** 2)))
        rmse_pace = float(np.sqrt(np.mean((p_pace - y) ** 2)))
        mae_ols = float(np.mean(ae_ols))
        mae_pace = float(np.mean(ae_pace))
        lo, hi = _cluster_boot_ci(d, ae_pace - ae_ols, n_boot=n_boot, seed=seed)

        def _fmt_bias(p):
            err = p - y
            b_all = f"{float(np.mean(err)):+.3f}"
            b_cl = f"{float(np.mean(err[sp <= 4.0])):+.3f}" if np.any(sp <= 4.0) else "n/a"
            b_bl = f"{float(np.mean(err[sp >= 10.0])):+.3f}" if np.any(sp >= 10.0) else "n/a"
            return b_all, b_cl, b_bl

        bo, bco, bbo = _fmt_bias(p_ols)
        bp, bcp, bbp = _fmt_bias(p_pace)
        lines.append(
            f"| {label} | ols | {n_games} | {n_days} | {mae_ols:.3f} | +0.000 | - "
            f"| {rmse_ols:.3f} | +0.000 | {bo} | {bco} | {bbo} |"
        )
        lines.append(
            f"| {label} | pace | {n_games} | {n_days} | {mae_pace:.3f} "
            f"| {mae_pace - mae_ols:+.3f} | [{lo:+.3f}, {hi:+.3f}] "
            f"| {rmse_pace:.3f} | {rmse_pace - rmse_ols:+.3f} | {bp} | {bcp} | {bbp} |"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", default=None, help="SQLite DB path (default: config.db_path())")
    ap.add_argument("--seasons", default=None,
                    help="comma-separated NBA season end-years to score (e.g. 2024,2025,2026,2027)")
    ap.add_argument("--refit", choices=("weekly", "daily"), default="weekly")
    ap.add_argument("--lam-pace", type=float, default=nba_totals.LAM_PACE)
    ap.add_argument("--lam-eff", type=float, default=nba_totals.LAM_EFF)
    ap.add_argument("--min-train", type=int, default=80,
                    help="minimum prior games with box-score possessions before scoring")
    ap.add_argument("--half-life", type=int, default=None,
                    help="decay half-life in days (default: weights.half_life_for('nba'))")
    ap.add_argument("--boot", type=int, default=2000, help="day-cluster bootstrap resamples")
    ap.add_argument("--seed", type=int, default=20261126)
    ap.add_argument("--use-stan", action="store_true",
                    help="use Stan Bradley-Terry ratings instead of fast ridge stand-in")
    args = ap.parse_args(argv)

    path = Path(args.db or db_path())
    if not path.is_file():
        print(f"database not found: {path}", file=sys.stderr)
        return 1
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    try:
        df = load_nba_games(conn)
    finally:
        conn.close()

    n_poss = int(df["poss"].notna().sum()) if "poss" in df.columns else 0
    n_reg = int(df["reg_home_score"].notna().sum()) if "reg_home_score" in df.columns else 0
    print(f"NBA completed games: {len(df)} | with poss: {n_poss} | with reg scores: {n_reg}")
    if n_poss < args.min_train:
        print("Too few games with box-score possessions; run `python -m bayesratings.team_counts --leagues nba` first.")
        return 1

    seasons = ([int(x.strip()) for x in args.seasons.split(",") if x.strip()]
               if args.seasons else None)
    scored = evaluate_walkforward(
        df, seasons=seasons, refit=args.refit,
        lam_pace=args.lam_pace, lam_eff=args.lam_eff,
        min_train=args.min_train, half_life_days=args.half_life,
        use_stan=args.use_stan,
    )
    print(f"refit={args.refit} lam_pace={args.lam_pace} lam_eff={args.lam_eff} "
          f"ratings={'stan' if args.use_stan else 'ridge'}\n")
    print(format_report(scored, n_boot=args.boot, seed=args.seed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
