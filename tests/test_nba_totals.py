"""NBA regulation/overtime pace-efficiency totals model tests."""

import gzip
import importlib.util
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import pytest

from bayesratings import config, db, nba_totals, team_counts
from bayesratings.forecast import forecast_games
from bayesratings.games import validate_games
from bayesratings.hybrid import fit_hybrid_models, points_model_for
from bayesratings.team_stats import compute_team_stats


def _espn_nba_summary(home_eid, away_eid, home_quarters, away_quarters,
                      home_fga=88, away_fga=86, game_note=None):
    """Minimal realistic ESPN NBA summary JSON for team_counts.parse / extract."""
    def box_side(eid, fga):
        return {
            "team": {"id": str(eid)},
            "statistics": [
                {"name": "fieldGoalsMade-fieldGoalsAttempted", "displayValue": f"40-{fga}"},
                {"name": "threePointFieldGoalsMade-threePointFieldGoalsAttempted", "displayValue": "12-34"},
                {"name": "freeThrowsMade-freeThrowsAttempted", "displayValue": "17-22"},
                {"name": "offensiveRebounds", "displayValue": "10"},
                {"name": "defensiveRebounds", "displayValue": "34"},
                {"name": "assists", "displayValue": "25"},
                {"name": "steals", "displayValue": "7"},
                {"name": "blocks", "displayValue": "5"},
                {"name": "totalTurnovers", "displayValue": "13"},
                {"name": "fouls", "displayValue": "19"},
                {"name": "fastBreakPoints", "displayValue": "14"},
                {"name": "pointsInPaint", "displayValue": "48"},
                {"name": "turnoverPoints", "displayValue": "15"},
            ],
        }

    header = {
        "competitions": [{
            "competitors": [
                {"id": str(home_eid), "team": {"id": str(home_eid)},
                 "linescores": home_quarters},
                {"id": str(away_eid), "team": {"id": str(away_eid)},
                 "linescores": away_quarters},
            ],
        }],
    }
    if game_note is not None:
        header["gameNote"] = game_note
    return {
        "header": header,
        "boxscore": {
            "teams": [box_side(home_eid, home_fga), box_side(away_eid, away_fga)],
            "players": [],
        },
    }


def test_team_counts_nba_reg_pts_and_schema_contract():
    # Existing indices 0..17 stay intact; reg_pts is appended at index 18.
    assert team_counts.VERSION >= 5
    nba_cols = team_counts.COUNTS["nba"]
    assert nba_cols[16] == "poss"
    assert nba_cols[17] == "minutes"
    assert nba_cols[18] == "reg_pts"

    # Regulation game with displayValue linescores (current ESPN format).
    reg_sum = _espn_nba_summary(
        1, 2,
        [{"displayValue": "28"}, {"displayValue": "30"}, {"displayValue": "26"}, {"displayValue": "29"}],
        [{"displayValue": "25"}, {"displayValue": "27"}, {"displayValue": "28"}, {"displayValue": "24"}],
    )
    parsed = team_counts.parse(reg_sum, "nba", ["1", "2"])
    assert parsed["1"][17] == 48 and parsed["2"][17] == 48
    assert parsed["1"][18] == 113 and parsed["2"][18] == 104
    assert parsed["1"][16] == parsed["2"][16] and 80 < parsed["1"][16] < 115

    # Overtime game (5 periods): reg_pts sums Q1-Q4 only, minutes == 53.
    ot_sum = _espn_nba_summary(
        3, 4,
        [{"displayValue": "35"}, {"displayValue": "19"}, {"displayValue": "29"},
         {"displayValue": "26"}, {"displayValue": "14"}],
        [{"displayValue": "31"}, {"displayValue": "25"}, {"displayValue": "23"},
         {"displayValue": "30"}, {"displayValue": "10"}],
        home_fga=98, away_fga=96,
    )
    parsed_ot = team_counts.parse(ot_sum, "nba", ["3", "4"])
    assert parsed_ot["3"][17] == 53 and parsed_ot["4"][17] == 53
    assert parsed_ot["3"][18] == 109 and parsed_ot["4"][18] == 109

    # Legacy numeric `value` linescores also parse; fewer than 4 quarters -> None.
    legacy = _espn_nba_summary(
        5, 6,
        [{"value": 30.0}, {"value": 25.0}, {"value": 28.0}, {"value": 22.0}],
        [{"value": 20.0}, {"value": 21.0}, {"value": 22.0}],
    )
    parsed_leg = team_counts.parse(legacy, "nba", ["5", "6"])
    assert parsed_leg["5"][18] == 105
    assert parsed_leg["6"][18] is None


def _synthetic_nba_frame(seed=11, n_games=600):
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(12)]
    # Clear pace and efficiency separation across teams
    tau = np.linspace(-5.0, 5.0, len(teams))
    off = np.linspace(-6.0, 6.0, len(teams))
    defn = np.linspace(5.0, -5.0, len(teams))  # higher defn = worse defense (allows more)
    rows = []
    start = pd.Timestamp("2026-10-20")
    for k in range(n_games):
        h, a = rng.choice(len(teams), 2, replace=False)
        neut = int(k % 40 == 0)
        p_reg = 99.0 + tau[h] + tau[a] + rng.normal(0.0, 2.5)
        hca = 0.0 if neut else 1.5
        eh = 114.0 + hca + off[h] + defn[a] + rng.normal(0.0, 6.0)
        ea = 114.0 - hca + off[a] + defn[h] + rng.normal(0.0, 6.0)
        rh = int(round(p_reg * eh / 100.0))
        ra = int(round(p_reg * ea / 100.0))
        is_ot = (rh == ra) or (abs(rh - ra) == 1 and rng.random() < 0.5)
        if is_ot:
            tie = rh
            rh = ra = tie
            fh, fa = rh + 12, ra + 10
            mins = 53.0
            poss_obs = p_reg + 11.0  # inflated full-game OT possessions
        else:
            fh, fa = rh, ra
            mins = 48.0
            poss_obs = p_reg
        rows.append({
            "season": 2027,
            "date": (start + pd.Timedelta(days=k // 6)).strftime("%Y-%m-%d"),
            "home_team": teams[h],
            "away_team": teams[a],
            "home_score": float(fh),
            "away_score": float(fa),
            "neutral": neut,
            "poss": float(poss_obs),
            "minutes": float(mins),
            "reg_home_score": float(rh),
            "reg_away_score": float(ra),
        })
    games = validate_games(pd.DataFrame(rows))
    truth = {"teams": teams, "tau": tau, "off": off, "defn": defn}
    return games, truth


def test_pace_efficiency_recovery_and_ot_isolation():
    games, truth = _synthetic_nba_frame(seed=11, n_games=700)
    # Inject extra OT games with huge full-game possessions for the slowest team (T00)
    # to confirm OT games do not contaminate its estimated 48-minute pace.
    extra_ot = []
    for k in range(15):
        extra_ot.append({
            "season": 2027,
            "date": f"2027-02-{1 + k:02d}",
            "home_team": "T00",
            "away_team": "T01",
            "home_score": 132.0,
            "away_score": 128.0,
            "neutral": 0,
            "poss": 116.0,
            "minutes": 58.0,
            "reg_home_score": 108.0,
            "reg_away_score": 108.0,
        })
    augmented = validate_games(pd.concat([games, pd.DataFrame(extra_ot)], ignore_index=True))
    fit = nba_totals.fit_nba_pace_totals(augmented)
    assert fit is not None
    assert fit.n_ot_games >= 15

    est_tau = np.array([fit.pace[t] for t in truth["teams"]])
    est_off = np.array([fit.off_eff[t] for t in truth["teams"]])
    est_def = np.array([fit.def_eff[t] for t in truth["teams"]])
    assert np.corrcoef(est_tau, truth["tau"])[0, 1] > 0.95
    assert np.corrcoef(est_off, truth["off"])[0, 1] > 0.90
    assert np.corrcoef(est_def, truth["defn"])[0, 1] > 0.90
    # Slowest team stays clearly slower than fastest team despite 15 2OT games.
    assert fit.pace["T00"] < -2.5 and fit.pace["T11"] > 2.5
    # Home-court efficiency edge is positive.
    assert fit.hca_eff > 0.5


def test_opponent_adjusted_sos_efficiency():
    # Team A and Team B score the exact same 110 points on 100 possessions,
    # but Team A plays elite defenses (D1, D2) while Team B plays porous
    # defenses (W1, W2). Opponent-adjusted Ridge must rate A's offense higher.
    rows = []
    day = pd.Timestamp("2026-11-01")
    k = 0
    def add(h, a, hs, as_):
        nonlocal k
        rows.append({
            "season": 2027,
            "date": (day + pd.Timedelta(days=k // 4)).strftime("%Y-%m-%d"),
            "home_team": h, "away_team": a,
            "home_score": float(hs), "away_score": float(as_),
            "neutral": 1, "poss": 100.0, "minutes": 48.0,
            "reg_home_score": float(hs), "reg_away_score": float(as_),
        })
        k += 1

    # Anchor D1/D2 as elite defenses (hold average teams C1/C2 to 98) and
    # W1/W2 as weak defenses (allow 122 to C1/C2).
    for _ in range(10):
        for c in ("C1", "C2"):
            for d in ("D1", "D2"):
                add(c, d, 98, 110)
            for w in ("W1", "W2"):
                add(c, w, 122, 110)
    # Team A scores 110 vs D1/D2; Team B scores 110 vs W1/W2.
    for _ in range(8):
        add("A", "D1", 110, 110)
        add("A", "D2", 110, 110)
        add("B", "W1", 110, 110)
        add("B", "W2", 110, 110)

    games = validate_games(pd.DataFrame(rows))
    fit = nba_totals.fit_nba_pace_totals(games)
    assert fit is not None
    assert fit.off_eff["A"] > fit.off_eff["B"] + 5.0
    assert fit.def_eff["D1"] < fit.def_eff["W1"] - 5.0


def test_margin_dependent_ot_and_hybrid_forecast_invariants():
    games, truth = _synthetic_nba_frame(seed=23, n_games=600)
    ratings = pd.Series(
        {t: (truth["off"][i] - truth["defn"][i]) / 10.0
         for i, t in enumerate(truth["teams"])}
    )
    stats = compute_team_stats(games)
    hyb_ols = fit_hybrid_models(games, ratings, stats["team_stats"], points_model="ols")
    hyb_pace = fit_hybrid_models(games, ratings, stats["team_stats"], points_model="pace")
    assert hyb_pace["pace_fit"] is not None

    # Close game receives strictly larger expected OT points than a blowout,
    # and OT lift is symmetric in margin sign.
    pf = hyb_pace["pace_fit"]
    assert not hasattr(pf, "predict")
    ot_close = pf.expected_ot_points([0.0, 2.0, -2.0])
    ot_blow = pf.expected_ot_points([15.0, -15.0])
    assert ot_close[0] > ot_blow[0] + 0.3
    assert abs(ot_close[1] - ot_close[2]) < 1e-12
    assert abs(ot_blow[0] - ot_blow[1]) < 1e-12

    sched = validate_games(pd.DataFrame([
        # Fast vs fast, close
        {"season": 2027, "date": "2027-03-01", "home_team": "T10", "away_team": "T11",
         "home_score": np.nan, "away_score": np.nan, "neutral": 0},
        # Slow vs slow, neutral
        {"season": 2027, "date": "2027-03-01", "home_team": "T00", "away_team": "T01",
         "home_score": np.nan, "away_score": np.nan, "neutral": 1},
        # Unknown expansion team degrades cleanly to league average
        {"season": 2027, "date": "2027-03-01", "home_team": "NEW_TEAM", "away_team": "T05",
         "home_score": np.nan, "away_score": np.nan, "neutral": 0},
    ]))
    swapped = sched.rename(columns={"home_team": "away_team", "away_team": "home_team"})

    fc_ols = forecast_games(sched, ratings, stats["team_stats"], hyb_ols, stats["league_avg_points"])
    fc_pace = forecast_games(sched, ratings, stats["team_stats"], hyb_pace, stats["league_avg_points"])
    fc_swap = forecast_games(swapped, ratings, stats["team_stats"], hyb_pace, stats["league_avg_points"])

    # Win probabilities, spreads, and odds must be bit-for-bit identical to OLS.
    for col in ("home_prob", "away_prob", "home_spread", "away_spread", "home_odds", "away_odds"):
        assert (fc_pace[col].to_numpy() == fc_ols[col].to_numpy()).all(), col

    # Fast vs fast projects strictly higher total than slow vs slow.
    assert fc_pace.loc[0, "total_pred"] > fc_pace.loc[1, "total_pred"] + 10.0
    # Totals differ from OLS and remain finite even for an unseen team.
    assert np.isfinite(fc_pace["total_pred"]).all()
    assert not np.allclose(fc_pace["total_pred"].to_numpy(), fc_ols["total_pred"].to_numpy())
    # Neutral-site swap symmetry and margin reconciliation.
    assert abs(fc_pace.loc[1, "home_total_pred"] - fc_swap.loc[1, "away_total_pred"]) < 0.11
    assert abs(fc_pace.loc[1, "total_pred"] - fc_swap.loc[1, "total_pred"]) < 0.11
    assert (fc_pace["total_pred"] - fc_pace["home_total_pred"] - fc_pace["away_total_pred"]).abs().max() < 0.11
    diff = fc_pace["home_total_pred"] - fc_pace["away_total_pred"]
    assert (diff + fc_pace["home_spread"]).abs().max() < 0.21


def test_sparse_or_missing_boxscores_fallback_to_ols():
    games, _ = _synthetic_nba_frame(seed=5, n_games=40)
    # Fewer than MIN_REG_GAMES regulation rows with valid possessions -> None.
    sparse = games.copy()
    sparse.loc[5:, "poss"] = np.nan
    assert nba_totals.fit_nba_pace_totals(sparse) is None
    assert nba_totals.fit_nba_pace_totals(games.drop(columns=["poss"])) is None


def test_db_extract_and_attach_nba_box_cols(tmp_path):
    db_file = tmp_path / "nba_totals.db"
    conn = db.connect(db_file)
    db.init_db(conn)
    ids = db.upsert_teams(conn, "nba", [
        {"espn_id": 10, "abbreviation": "ATL", "name": "Hawks"},
        {"espn_id": 20, "abbreviation": "BOS", "name": "Celtics"},
        {"espn_id": 30, "abbreviation": "CLE", "name": "Cavaliers"},
        {"espn_id": 40, "abbreviation": "DEN", "name": "Nuggets"},
    ])
    # Insert 10 finished games so no team is treated as exhibition (MIN_TEAM_GAMES=5).
    game_rows = []
    summaries = []
    for i in range(10):
        eid = 9000 + i
        is_ot = (i == 1)
        is_cup_final = (i == 2)
        h_e, a_e = (10, 20) if i % 2 == 0 else (30, 40)
        if is_ot:
            hq = [{"displayValue": "28"}, {"displayValue": "27"},
                  {"displayValue": "26"}, {"displayValue": "28"}, {"displayValue": "14"}]
            aq = [{"displayValue": "25"}, {"displayValue": "29"},
                  {"displayValue": "27"}, {"displayValue": "28"}, {"displayValue": "10"}]
            hs, as_ = 123, 119
        else:
            hq = [{"displayValue": "28"}, {"displayValue": "29"},
                  {"displayValue": "27"}, {"displayValue": "26"}]
            aq = [{"displayValue": "24"}, {"displayValue": "26"},
                  {"displayValue": "25"}, {"displayValue": "27"}]
            hs, as_ = 110, 102
        game_rows.append({
            "espn_event_id": eid, "league": "nba", "season": 2027, "season_type": 2,
            "start_time_utc": f"2026-11-{10 + i:02d}T00:00:00Z",
            "game_date": f"2026-11-{10 + i:02d}",
            "home_team_id": ids[h_e], "away_team_id": ids[a_e],
            "home_score": hs, "away_score": as_, "completed": 1,
            "status": "STATUS_FINAL", "venue": "Arena", "neutral_site": 0,
            "covid_2020": 0, "if_necessary": 0, "series_completed": 0,
        })
        if i != 9:  # leave game 9 to insert a legacy v4 team_game_stats row manually
            summ = _espn_nba_summary(
                h_e, a_e, hq, aq,
                game_note=("NBA Cup Championship" if is_cup_final else None),
            )
            summaries.append((eid, "nba", "2026-11-26T00:00:00Z",
                              gzip.compress(json.dumps(summ).encode())))

    db.upsert_games(conn, game_rows)
    with db.immediate(conn):
        conn.executemany(
            "INSERT INTO game_summaries (espn_event_id, league, fetched_at, json_gz) "
            "VALUES (?, ?, ?, ?)", summaries)

    res = team_counts.extract(conn, leagues=("nba",), log=lambda *_: None)
    assert res["nba"]["games"] == 9

    # Insert a legacy v4 row (18 entries, no reg_pts) for event 9009 to verify
    # backward compatibility before a full team_counts re-extraction runs.
    v4_counts = [0] * 16 + [98.5, 48]
    with db.immediate(conn):
        conn.execute(
            "INSERT INTO team_game_stats (espn_event_id, team_id, opp_team_id, league, "
            "season, season_type, game_date, is_home, points_for, points_against, "
            "summary_fetched_at, version, counts_json) "
            "VALUES (?, ?, ?, 'nba', 2027, 2, '2026-11-19', 1, 110, 102, "
            "'2026-11-20T00:00:00Z', 4, ?)",
            (9009, ids[30], ids[40], json.dumps([v4_counts, v4_counts])),
        )

    # 1. Join via espn_event_id (daily & replay path).
    df_id = db.read_games(conn, "nba", event_id=True)
    enriched = nba_totals.attach_nba_box_cols(conn, df_id)
    by_eid = enriched.set_index("espn_event_id")
    # Regulation game (9000)
    assert by_eid.loc[9000, "minutes"] == 48.0
    assert by_eid.loc[9000, "reg_home_score"] == 110.0
    assert by_eid.loc[9000, "reg_away_score"] == 102.0
    assert 80.0 < by_eid.loc[9000, "poss"] < 115.0
    # OT game (9001): reg scores 109-109, minutes 53
    assert by_eid.loc[9001, "minutes"] == 53.0
    assert by_eid.loc[9001, "reg_home_score"] == 109.0
    assert by_eid.loc[9001, "reg_away_score"] == 109.0
    # NBA Cup Championship (9002) has season_type 0 in team_game_stats -> excluded
    assert pd.isna(by_eid.loc[9002, "poss"])
    # Legacy v4 regulation row (9009) gets poss=98.5, minutes=48, reg scores from final
    assert by_eid.loc[9009, "poss"] == 98.5
    assert by_eid.loc[9009, "minutes"] == 48.0
    assert by_eid.loc[9009, "reg_home_score"] == 110.0
    assert by_eid.loc[9009, "reg_away_score"] == 102.0

    # 2. Fallback join without espn_event_id column.
    df_noid = db.read_games(conn, "nba", event_id=False)
    enriched_noid = nba_totals.attach_nba_box_cols(conn, df_noid)
    ot_row = enriched_noid[enriched_noid["date"] == "2026-11-11"].iloc[0]
    assert ot_row["minutes"] == 53.0
    assert ot_row["reg_home_score"] == 109.0
    assert ot_row["reg_away_score"] == 109.0
    conn.close()


def test_nba_totals_env_toggle(monkeypatch):
    monkeypatch.delenv("BAYESRATINGS_NBA_TOTALS", raising=False)
    assert config.nba_points_model() == "ols"
    assert points_model_for("nba") == "ols"

    monkeypatch.setenv("BAYESRATINGS_NBA_TOTALS", "pace")
    assert config.nba_points_model() == "pace"
    assert points_model_for("nba") == "pace"

    monkeypatch.setenv("BAYESRATINGS_NBA_TOTALS", "OLS")
    assert points_model_for("nba") == "ols"

    monkeypatch.setenv("BAYESRATINGS_NBA_TOTALS", "bogus")
    with pytest.raises(ValueError):
        points_model_for("nba")


def test_daily_attaches_nba_box_cols_when_pace_enabled(monkeypatch):
    from bayesratings import daily

    seen = {}

    class Stop(Exception):
        pass

    def fake_run_model(done, **kw):
        seen["cols"] = set(done.columns)
        seen["points_model"] = kw.get("points_model")
        raise Stop

    games = pd.DataFrame({
        "espn_event_id": [1],
        "season": [2027],
        "date": ["2026-11-10"],
        "home_team": ["1"],
        "away_team": ["2"],
        "home_score": [110.0],
        "away_score": [105.0],
        "neutral": [0],
    })

    def fake_attach(_conn, df):
        out = df.copy()
        out["poss"] = 99.0
        out["minutes"] = 48.0
        out["reg_home_score"] = 110.0
        out["reg_away_score"] = 105.0
        return out

    monkeypatch.setattr(daily.db, "read_games", lambda conn, league, event_id=False: games)
    monkeypatch.setattr(daily.nba_totals, "attach_nba_box_cols", fake_attach)
    monkeypatch.setattr(daily, "todays_schedule",
                        lambda *a, **k: pd.DataFrame({"espn_event_id": []}))
    monkeypatch.setattr(daily, "run_model", fake_run_model)

    dummy_conn = sqlite3.connect(":memory:")
    try:
        # Default (unset -> ols): does not require pace columns
        monkeypatch.delenv("BAYESRATINGS_NBA_TOTALS", raising=False)
        try:
            daily.rate_league(dummy_conn, "nba", pd.Timestamp("2026-11-15").date(), None)
        except Stop:
            pass
        assert seen["points_model"] == "ols"
        assert "poss" not in seen["cols"]

        # Opt-in (pace): attaches box columns and passes points_model="pace"
        monkeypatch.setenv("BAYESRATINGS_NBA_TOTALS", "pace")
        try:
            daily.rate_league(dummy_conn, "nba", pd.Timestamp("2026-11-15").date(), None)
        except Stop:
            pass
    finally:
        dummy_conn.close()

    assert points_model_for("nba") == "pace"
    assert seen["points_model"] == "pace"
    assert {"poss", "minutes", "reg_home_score", "reg_away_score"} <= seen["cols"]
    assert "espn_event_id" not in seen["cols"]


def test_eval_nba_totals_script_smoke():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "eval_nba_totals.py"
    spec = importlib.util.spec_from_file_location("eval_nba_totals", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    games, _ = _synthetic_nba_frame(seed=17, n_games=180)
    scored = mod.evaluate_walkforward(games, refit="weekly", min_train=60)
    assert len(scored) > 20
    assert {"season", "date", "actual", "ols_pred", "pace_pred"} <= set(scored.columns)
    assert np.isfinite(scored["ols_pred"]).all() and np.isfinite(scored["pace_pred"]).all()
    table = mod.format_report(scored, n_boot=100, seed=1)
    assert "| ols |" in table and "| pace |" in table
