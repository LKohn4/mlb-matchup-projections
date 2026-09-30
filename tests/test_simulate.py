import numpy as np
import pytest

from matchup.simulate import StaffPlan, simulate_games

LEAGUE = np.array([0.226, 0.088, 0.142, 0.045, 0.004, 0.031, 0.464])


def grid(p):
    p = np.asarray(p, dtype=float)
    return np.broadcast_to(p / p.sum(), (2, 9, 6, 4, 7)).copy()


def test_league_average_scores_like_mlb():
    res = simulate_games(grid(LEAGUE), (StaffPlan(), StaffPlan()), n_sims=5000)
    per_team = res.game_summary()["total_runs"] / 2
    assert 3.9 < per_team < 4.9
    pa = res.batter_table().query("team == 0")["pa"].to_numpy()
    assert np.all(np.diff(pa) < 0)  # leadoff hitters bat most
    assert 4.4 < pa[0] < 5.1


def test_strikeout_team_never_scores_and_outs_are_complete():
    probs = grid(LEAGUE)
    probs[0] = 0
    probs[0, ..., 0] = 1  # away team always strikes out
    res = simulate_games(probs, (StaffPlan(), StaffPlan()), n_sims=500)
    assert res.score[:, 0].max() == 0
    assert (res.batting[:, 0].sum(axis=(1, 2)) % 3 == 0).all()  # three outs per half-inning
    assert res.game_summary()["home_win_prob"] == 1.0


def test_home_team_skips_bottom_ninth_when_ahead():
    probs = grid(LEAGUE)
    probs[0] = 0
    probs[0, ..., 0] = 1  # away never scores, so home wins every game
    res = simulate_games(probs, (StaffPlan(), StaffPlan()), n_sims=500)
    regulation = res.innings == 9
    outs_vs_home = res.pitch_outs[regulation, 0].sum(axis=1)  # away staff pitches to home
    assert outs_vs_home.max() <= 26  # never a full bottom of the 9th
    assert (outs_vs_home == 24).mean() > 0.8  # usually skipped entirely


def test_bullpen_usage_follows_plan():
    plan = StaffPlan(has_opener=True, opener_outs=3, starter_outs_mean=15,
                     reliever_weights=(1.0, 0.0, 0.0), has_closer=False)
    res = simulate_games(grid(LEAGUE), (plan, plan), n_sims=2000)
    pt = res.pitcher_table().query("staff == 0").set_index("role")
    assert pt.loc["Opener", "p_appear"] == 1.0
    assert pt.loc["Opener", "ip"] == pytest.approx(1.0, abs=0.05)
    assert pt.loc["Reliever 2", "p_appear"] == 0
    assert pt.loc["Closer", "p_appear"] == 0
    assert pt.loc["Reliever 1", "p_appear"] > 0.9


def test_reproducible_with_seed():
    a = simulate_games(grid(LEAGUE), (StaffPlan(), StaffPlan()), n_sims=200, seed=7)
    b = simulate_games(grid(LEAGUE), (StaffPlan(), StaffPlan()), n_sims=200, seed=7)
    np.testing.assert_array_equal(a.score, b.score)
