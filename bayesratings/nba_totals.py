"""NBA regulation/overtime pace-efficiency totals model.

Replaces the raw-PPG OLS/OS*DS blend for NBA expected points. Win
probabilities and spreads do not change.

Raw final-score PPG has two issues in NBA:
1. About 6% of games go to overtime (+~22.5 points, +~10.5 possessions).
   Inside a 90-day window, recent OT games skew team scoring averages, and the
   OLS intercept spreads the OT bump evenly across close games and blowouts.
2. Both teams share the same game possessions. Per-game OS/DS mixes tempo with
   efficiency and ignores opponent strength.

On the same time-decay window and weights as the ratings:
1. Weighted Ridge on non-OT games for 48-minute regulation possessions:
   ``poss = mu_pace + tau[home] + tau[away]``.
2. Possession-weighted Ridge on non-OT games for regulation offensive and
   defensive efficiency (points per 100 possessions):
   ``eff_home = mu_eff + hca_eff * (1 - neutral) + off[home] + def[away]``
   ``eff_away = mu_eff - hca_eff * (1 - neutral) + off[away] + def[home]``
3. Margin-dependent overtime lift fit from the training window:
   ``p_ot(m) = clip(kappa * [Phi((0.5 - m) / sigma_m) - Phi((-0.5 - m) / sigma_m)], 0, 0.25)``
   ``E[OT_pts | m] = p_ot(m) * mu_ot``

Returns ``None`` when `poss` is missing or fewer than ``MIN_REG_GAMES`` non-OT
games have valid possessions, so callers fall back to the OLS blend.
"""

import json
import math
import sqlite3

import numpy as np
import pandas as pd

from . import team_counts

# Empirical-Bayes Ridge penalties (sigma_noise^2 / sigma_team^2) calibrated for
# the 90-day NBA time-decay window.
LAM_PACE = 3.0
LAM_EFF = 8.0

# Minimum regulation games with valid possessions required to fit the pace model.
MIN_REG_GAMES = 15

# Sanity bounds on 48-minute NBA regulation possessions.
MIN_POSS = 75.0
MAX_POSS = 125.0

# Empirical-Bayes shrinkage priors for overtime frequency and extra OT points.
PRIOR_OT_RATE = 0.06
PRIOR_OT_RATE_WEIGHT = 20.0
PRIOR_OT_PTS = 22.5
PRIOR_OT_PTS_WEIGHT = 10.0
MAX_OT_PROB = 0.25

_NBA_COUNTS = team_counts.COUNTS["nba"]
_POSS_IDX = _NBA_COUNTS.index("poss")
_MIN_IDX = _NBA_COUNTS.index("minutes")
_REG_PTS_IDX = _NBA_COUNTS.index("reg_pts")


def _norm_cdf(x):
    """Standard normal CDF without a scipy dependency."""
    arr = np.asarray(x, dtype=float)
    flat = [0.5 * (1.0 + math.erf(float(v) / math.sqrt(2.0)))
            for v in arr.ravel()]
    return np.asarray(flat, dtype=float).reshape(arr.shape)


def _to_float(v):
    if v is None or isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def attach_nba_box_cols(conn, games: pd.DataFrame) -> pd.DataFrame:
    """Attach `poss`, `minutes`, `reg_home_score`, `reg_away_score` from
    `team_game_stats` onto a completed NBA games frame.

    Matches by `espn_event_id` when present on `games`, else by
    `(game_date, home_team, away_team)`. Supports both `team_counts` v5 rows
    (which include `reg_pts`) and legacy v4 rows (regulation scores inferred
    where `minutes <= 48.5`). Returns `games` unchanged if `conn` is None or
    `team_game_stats` has no NBA rows.
    """
    if conn is None or games is None or len(games) == 0:
        return games
    try:
        rows = conn.execute(
            "SELECT espn_event_id, game_date, team_id, opp_team_id, "
            "points_for, points_against, counts_json "
            "FROM team_game_stats "
            "WHERE league = 'nba' AND is_home = 1 "
            "AND COALESCE(season_type, 2) IN (2, 3)"
        ).fetchall()
    except sqlite3.Error:
        return games
    if not rows:
        return games

    by_eid = {}
    by_key = {}
    for row in rows:
        is_map = isinstance(row, sqlite3.Row)
        eid = row["espn_event_id"] if is_map else row[0]
        gdate = str(row["game_date"] if is_map else row[1])[:10]
        hid = str(row["team_id"] if is_map else row[2])
        aid = str(row["opp_team_id"] if is_map else row[3])
        pf = _to_float(row["points_for"] if is_map else row[4])
        pa = _to_float(row["points_against"] if is_map else row[5])
        raw_json = row["counts_json"] if is_map else row[6]
        if not raw_json:
            continue
        try:
            counts = json.loads(raw_json)
            home_c = counts[0]
            away_c = counts[1] if len(counts) > 1 else None
        except (TypeError, ValueError, IndexError, json.JSONDecodeError):
            continue
        if not isinstance(home_c, list) or len(home_c) <= _MIN_IDX:
            continue
        poss = _to_float(home_c[_POSS_IDX])
        minutes = _to_float(home_c[_MIN_IDX])
        reg_h = (_to_float(home_c[_REG_PTS_IDX])
                 if len(home_c) > _REG_PTS_IDX else None)
        reg_a = (_to_float(away_c[_REG_PTS_IDX])
                 if isinstance(away_c, list) and len(away_c) > _REG_PTS_IDX
                 else None)
        # Legacy v4 fallback: regulation games have minutes <= 48.5, so final
        # score equals regulation score.
        if reg_h is None and reg_a is None and minutes is not None and minutes <= 48.5:
            reg_h, reg_a = pf, pa
        rec = (poss, minutes, reg_h, reg_a)
        if eid is not None:
            by_eid[int(eid)] = rec
        by_key[(gdate, hid, aid)] = rec

    if not by_eid and not by_key:
        return games

    n = len(games)
    poss_col = np.full(n, np.nan, dtype=float)
    min_col = np.full(n, np.nan, dtype=float)
    rh_col = np.full(n, np.nan, dtype=float)
    ra_col = np.full(n, np.nan, dtype=float)

    has_eid = "espn_event_id" in games.columns
    eids = games["espn_event_id"].to_numpy() if has_eid else None
    dates = pd.to_datetime(games["date"]).dt.strftime("%Y-%m-%d").to_numpy()
    homes = games["home_team"].astype(str).to_numpy()
    aways = games["away_team"].astype(str).to_numpy()

    for i in range(n):
        rec = None
        if has_eid and pd.notna(eids[i]):
            try:
                rec = by_eid.get(int(eids[i]))
            except (TypeError, ValueError):
                rec = None
        if rec is None:
            rec = by_key.get((dates[i], homes[i], aways[i]))
        if rec is None:
            continue
        p, m, rh, ra = rec
        if p is not None:
            poss_col[i] = p
        if m is not None:
            min_col[i] = m
        if rh is not None:
            rh_col[i] = rh
        if ra is not None:
            ra_col[i] = ra

    out = games.copy()
    out["poss"] = poss_col
    out["minutes"] = min_col
    out["reg_home_score"] = rh_col
    out["reg_away_score"] = ra_col
    return out


class _PaceTotalsModel:
    """Fitted opponent-adjusted regulation pace-efficiency + OT totals model."""

    def __init__(self, teams, mu_pace, tau, mu_eff, hca_eff, off_eff, def_eff,
                 p_ot_emp, mu_ot, sigma_margin, kappa_ot,
                 n_reg_games, n_ot_games):
        self.teams = list(teams)
        self._t_idx = {t: i for i, t in enumerate(self.teams)}
        self.mu_pace = float(mu_pace)
        self._tau = np.asarray(tau, dtype=float)
        self.pace = pd.Series(self._tau, index=self.teams, name="pace")
        self.mu_eff = float(mu_eff)
        self.hca_eff = float(hca_eff)
        self._off = np.asarray(off_eff, dtype=float)
        self._def = np.asarray(def_eff, dtype=float)
        self.off_eff = pd.Series(self._off, index=self.teams, name="off_eff")
        self.def_eff = pd.Series(self._def, index=self.teams, name="def_eff")
        self.p_ot_emp = float(p_ot_emp)
        self.mu_ot = float(mu_ot)
        self.sigma_margin = float(sigma_margin)
        self.kappa_ot = float(kappa_ot)
        self.n_reg_games = int(n_reg_games)
        self.n_ot_games = int(n_ot_games)

    def _lookup(self, team_col, vec):
        t_idx = self._t_idx
        return np.array([vec[t_idx[str(t)]] if str(t) in t_idx else 0.0
                         for t in team_col], dtype=float)

    def predict_regulation(self, schedule: pd.DataFrame):
        """Expected 48-minute regulation points (home, away) before OT lift."""
        h_tau = self._lookup(schedule["home_team"], self._tau)
        a_tau = self._lookup(schedule["away_team"], self._tau)
        h_off = self._lookup(schedule["home_team"], self._off)
        a_off = self._lookup(schedule["away_team"], self._off)
        h_def = self._lookup(schedule["home_team"], self._def)
        a_def = self._lookup(schedule["away_team"], self._def)
        neut = (schedule["neutral"].fillna(0).to_numpy(dtype=float)
                if "neutral" in schedule.columns
                else np.zeros(len(schedule), dtype=float))

        p_hat = np.clip(self.mu_pace + h_tau + a_tau, MIN_POSS, MAX_POSS)
        eh_hat = self.mu_eff + self.hca_eff * (1.0 - neut) + h_off + a_def
        ea_hat = self.mu_eff - self.hca_eff * (1.0 - neut) + a_off + h_def
        pts_h = np.maximum(p_hat * eh_hat / 100.0, 0.0)
        pts_a = np.maximum(p_hat * ea_hat / 100.0, 0.0)
        return pts_h, pts_a

    def expected_ot_points(self, margin_pred) -> np.ndarray:
        """Expected total overtime points as an even function of predicted margin."""
        m = np.asarray(margin_pred, dtype=float)
        p_tie = (_norm_cdf((0.5 - m) / self.sigma_margin)
                 - _norm_cdf((-0.5 - m) / self.sigma_margin))
        p_ot = np.clip(self.kappa_ot * p_tie, 0.0, MAX_OT_PROB)
        return p_ot * self.mu_ot


def fit_nba_pace_totals(games: pd.DataFrame, sample_weight=None,
                        lam_pace: float = LAM_PACE,
                        lam_eff: float = LAM_EFF):
    """Fit opponent-adjusted NBA regulation pace-efficiency + OT model.

    Expects `poss` and either `minutes` or (`reg_home_score`, `reg_away_score`)
    columns on `games`. Returns ``None`` if `poss` is missing or fewer than
    ``MIN_REG_GAMES`` valid regulation games are available, allowing callers to
    fall back to the OLS totals blend.
    """
    if games is None or len(games) == 0 or "poss" not in games.columns:
        return None

    n = len(games)
    w = (np.ones(n, dtype=float) if sample_weight is None
         else np.asarray(sample_weight, dtype=float))
    if len(w) != n:
        raise ValueError("sample_weight must match games row for row.")

    poss = pd.to_numeric(games["poss"], errors="coerce").to_numpy(dtype=float)
    mins = (pd.to_numeric(games["minutes"], errors="coerce").to_numpy(dtype=float)
            if "minutes" in games.columns else np.full(n, np.nan, dtype=float))
    hs = pd.to_numeric(games["home_score"], errors="coerce").to_numpy(dtype=float)
    as_ = pd.to_numeric(games["away_score"], errors="coerce").to_numpy(dtype=float)
    rhs = (pd.to_numeric(games["reg_home_score"], errors="coerce").to_numpy(dtype=float)
           if "reg_home_score" in games.columns else np.full(n, np.nan, dtype=float))
    ras = (pd.to_numeric(games["reg_away_score"], errors="coerce").to_numpy(dtype=float)
           if "reg_away_score" in games.columns else np.full(n, np.nan, dtype=float))
    neut = (pd.to_numeric(games["neutral"], errors="coerce").fillna(0).to_numpy(dtype=float)
            if "neutral" in games.columns else np.zeros(n, dtype=float))

    has_reg = np.isfinite(rhs) & np.isfinite(ras)
    has_min = np.isfinite(mins)
    is_ot = ((has_min & (mins > 48.5))
             | (has_reg & ((hs + as_) > (rhs + ras) + 0.5)))

    valid_reg = (
        (~is_ot)
        & np.isfinite(poss)
        & (poss >= MIN_POSS)
        & (poss <= MAX_POSS)
        & np.isfinite(hs)
        & np.isfinite(as_)
        & (w > 0)
    )
    idx = np.flatnonzero(valid_reg)
    if len(idx) < MIN_REG_GAMES:
        return None

    home_s = games["home_team"].astype(str).to_numpy()
    away_s = games["away_team"].astype(str).to_numpy()
    teams = sorted(set(home_s) | set(away_s))
    t_idx = {t: i for i, t in enumerate(teams)}
    t_count = len(teams)
    h_idx = np.array([t_idx[t] for t in home_s], dtype=int)
    a_idx = np.array([t_idx[t] for t in away_s], dtype=int)

    # 1. Joint weighted Ridge for regulation possessions:
    #    poss = mu_pace + tau[home] + tau[away]
    wp = w[idx]
    yp = poss[idx]
    k = len(idx)
    rows_ar = np.arange(k)
    xp = np.zeros((k, 1 + t_count), dtype=float)
    xp[:, 0] = 1.0
    xp[rows_ar, 1 + h_idx[idx]] += 1.0
    xp[rows_ar, 1 + a_idx[idx]] += 1.0
    pp = np.zeros((1 + t_count, 1 + t_count), dtype=float)
    pp[1:, 1:] = float(lam_pace) * np.eye(t_count)
    sol_p = np.linalg.solve((xp.T * wp) @ xp + pp, (xp.T * wp) @ yp)
    tau = sol_p[1:]
    tau_mean = float(np.mean(tau))
    tau = tau - tau_mean
    mu_pace = float(sol_p[0] + 2.0 * tau_mean)

    # 2. Joint possession-weighted Ridge for regulation efficiency (pts / 100 poss):
    #    eff_h = mu_eff + hca * (1 - neutral) + off[home] + def[away]
    #    eff_a = mu_eff - hca * (1 - neutral) + off[away] + def[home]
    # Weighting by w * (poss / mu_pace) gives exact point conservation across
    # the weighted regulation sample.
    rh_use = np.where(has_reg, rhs, hs)[idx]
    ra_use = np.where(has_reg, ras, as_)[idx]
    eff_h = 100.0 * rh_use / yp
    eff_a = 100.0 * ra_use / yp
    we_game = wp * (yp / max(mu_pace, 1.0))
    nh = 1.0 - neut[idx]

    xe = np.zeros((2 * k, 2 + 2 * t_count), dtype=float)
    ye = np.empty(2 * k, dtype=float)
    we = np.concatenate([we_game, we_game])

    xe[:k, 0] = 1.0
    xe[:k, 1] = nh
    xe[rows_ar, 2 + h_idx[idx]] = 1.0
    xe[rows_ar, 2 + t_count + a_idx[idx]] = 1.0
    ye[:k] = eff_h

    xe[k:, 0] = 1.0
    xe[k:, 1] = -nh
    xe[k + rows_ar, 2 + a_idx[idx]] = 1.0
    xe[k + rows_ar, 2 + t_count + h_idx[idx]] = 1.0
    ye[k:] = eff_a

    pe = np.zeros((2 + 2 * t_count, 2 + 2 * t_count), dtype=float)
    pe[1, 1] = 1e-4  # safeguard if all training rows are neutral
    pe[2:, 2:] = float(lam_eff) * np.eye(2 * t_count)
    sol_e = np.linalg.solve((xe.T * we) @ xe + pe, (xe.T * we) @ ye)
    off = sol_e[2:2 + t_count]
    defn = sol_e[2 + t_count:]
    off_mean = float(np.mean(off))
    def_mean = float(np.mean(defn))
    off = off - off_mean
    defn = defn - def_mean
    mu_eff = float(sol_e[0] + off_mean + def_mean)
    hca_eff = float(sol_e[1])

    # 3. Empirical-Bayes overtime frequency, extra OT points, and margin-dependent
    #    tie calibration (strictly from the training window).
    known_ot = (has_min | has_reg) & np.isfinite(hs) & np.isfinite(as_) & (w > 0)
    w_known = float(np.sum(w[known_ot]))
    ot_known = is_ot & known_ot
    w_ot = float(np.sum(w[ot_known]))
    p_ot_emp = ((w_ot + PRIOR_OT_RATE_WEIGHT * PRIOR_OT_RATE)
                / (w_known + PRIOR_OT_RATE_WEIGHT))

    has_reg_ot = ot_known & has_reg
    if np.any(has_reg_ot):
        ot_pts = ((hs[has_reg_ot] + as_[has_reg_ot])
                  - (rhs[has_reg_ot] + ras[has_reg_ot]))
        ot_w = w[has_reg_ot]
        mu_ot = float((np.sum(ot_w * ot_pts) + PRIOR_OT_PTS_WEIGHT * PRIOR_OT_PTS)
                      / (np.sum(ot_w) + PRIOR_OT_PTS_WEIGHT))
    elif np.any(ot_known & has_min):
        m_ot = ot_known & has_min
        ot_pts = ((hs[m_ot] + as_[m_ot])
                  * np.maximum(mins[m_ot] - 48.0, 5.0)
                  / np.maximum(mins[m_ot], 53.0))
        ot_w = w[m_ot]
        mu_ot = float((np.sum(ot_w * ot_pts) + PRIOR_OT_PTS_WEIGHT * PRIOR_OT_PTS)
                      / (np.sum(ot_w) + PRIOR_OT_PTS_WEIGHT))
    else:
        mu_ot = PRIOR_OT_PTS

    # Residual SD of regulation margins around fitted regulation margin, used to
    # convert expected margin into regulation tie probability.
    reg_margin = np.where(has_reg, rhs - ras, hs - as_)
    tr_p = np.clip(mu_pace + tau[h_idx] + tau[a_idx], MIN_POSS, MAX_POSS)
    tr_m = tr_p * (2.0 * hca_eff * (1.0 - neut)
                   + (off[h_idx] + defn[a_idx])
                   - (off[a_idx] + defn[h_idx])) / 100.0
    reg_known = (~is_ot) & known_ot
    if np.sum(reg_known) >= MIN_REG_GAMES:
        resid = reg_margin[reg_known] - tr_m[reg_known]
        rw = w[reg_known]
        sigma_m = float(np.clip(
            np.sqrt(np.sum(rw * resid ** 2) / np.sum(rw)), 8.0, 20.0))
    else:
        sigma_m = 12.5

    if w_known > 0:
        raw_tie = (_norm_cdf((0.5 - tr_m[known_ot]) / sigma_m)
                   - _norm_cdf((-0.5 - tr_m[known_ot]) / sigma_m))
        mean_raw_tie = float(np.sum(w[known_ot] * raw_tie) / w_known)
        kappa_ot = float(np.clip(p_ot_emp / max(mean_raw_tie, 1e-4), 1.0, 3.5))
    else:
        kappa_ot = 1.8

    return _PaceTotalsModel(
        teams=teams, mu_pace=mu_pace, tau=tau,
        mu_eff=mu_eff, hca_eff=hca_eff, off_eff=off, def_eff=defn,
        p_ot_emp=p_ot_emp, mu_ot=mu_ot, sigma_margin=sigma_m,
        kappa_ot=kappa_ot, n_reg_games=k, n_ot_games=int(np.sum(ot_known)),
    )
