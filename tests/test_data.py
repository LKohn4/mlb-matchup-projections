import numpy as np
import pandas as pd
import pytest

from matchup.config import EVENT_TO_OUTCOME, OUTCOMES
from matchup.data import prepare, sample_games, validate_columns


def test_one_row_per_plate_appearance(season):
    assert not season.pa_all.duplicated(["game_pk", "at_bat_number"]).any()
    assert len(season.pa) < len(season.pa_all)  # intentional walks are counted, not modeled


def test_outcome_mapping(season):
    pa = season.pa
    assert pa["outcome"].between(0, len(OUTCOMES) - 1).all()
    expected = pa["events"].map(EVENT_TO_OUTCOME)
    assert (expected == pa["outcome"]).all()
    assert (pa.loc[pa["events"] == "home_run", "tb"] == 4).all()
    assert (pa.loc[pa["events"] == "hit_by_pitch", "on_base"] == 1).all()


def test_times_through_order_and_pa_number(season):
    pa = season.pa_all
    g = pa.loc[pa["game_pk"] == pa["game_pk"].iloc[0]].sort_values("at_bat_number")
    first_pitcher = g["pitcher"].iloc[0]
    tto = g.loc[g["pitcher"] == first_pitcher, "tto"].to_numpy()
    assert tto[0] == 1 and np.all(np.diff(tto) >= 0)
    assert (tto[:9] == 1).all() and (len(tto) <= 9 or tto[9] == 2)
    first_batter = g["batter"].iloc[0]
    assert g.loc[g["batter"] == first_batter, "batter_pa_num"].iloc[:3].tolist() == [1, 2, 3]


def test_pitch_count_is_before_the_plate_appearance(season):
    pa = season.pa_all.sort_values(["game_pk", "at_bat_number"])
    first = pa.groupby(["game_pk", "pitcher"]).head(1)
    assert (first["pitcher_pitch_count"] == 0).all()
    assert (pa["pitcher_pitch_count"] >= 0).all()


def test_day_night_mapping(raw):
    games = raw["game_pk"].unique()
    mapping = pd.Series(1.0, index=games[:10])
    sd = prepare(raw, mapping)
    assert set(sd.pa_all["is_night"].unique()) <= {0.5, 1.0}
    assert (sd.pa_all.loc[sd.pa_all["game_pk"].isin(games[:10]), "is_night"] == 1.0).all()


def test_validate_columns_names_missing(raw):
    with pytest.raises(ValueError, match="pitch_type"):
        validate_columns(raw.drop(columns=["pitch_type"]))


def test_sample_games_keeps_whole_games(raw):
    s = sample_games(raw, 0.3)
    kept = s["game_pk"].unique()
    assert 0 < len(kept) < raw["game_pk"].nunique()
    assert len(s) == raw["game_pk"].isin(kept).sum()


def test_untracked_batted_balls_are_not_hard_hit(raw):
    """Real Statcast has in-play pitches with no exit velocity (stored as <NA>)."""
    assert raw["launch_speed"].dtype.name == "Float64"  # same nullable type as pybaseball
    sd = prepare(raw)
    assert sd.pa["hard"].isin([0, 1]).all()
    untracked = sd.pa["bbe"].eq(1) & sd.pa["launch_speed"].isna()
    assert untracked.any() and (sd.pa.loc[untracked, "hard"] == 0).all()


def test_duplicate_pitch_rows_are_ignored(raw, season):
    doubled = pd.concat([raw, raw.head(500)], ignore_index=True)
    sd = prepare(doubled)
    assert len(sd.pitches) == len(season.pitches)
    assert len(sd.pa_all) == len(season.pa_all)
