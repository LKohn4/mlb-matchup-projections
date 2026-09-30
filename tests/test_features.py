import numpy as np
import pandas as pd

from matchup.data import prepare
from matchup.features import FEATURE_GROUPS, FEATURES, QUERY_COLUMNS, FeatureBuilder


def test_no_future_leakage(raw):
    """Features on or before a date don't change when later games are added."""
    cut = pd.Timestamp("2026-06-15")
    ref = cut + pd.Timedelta(days=1)
    full, early = prepare(raw), prepare(raw.loc[pd.to_datetime(raw["game_date"]) <= cut])
    a = FeatureBuilder(full, ref).transform(full.pa.loc[full.pa["game_date"] <= cut, list(QUERY_COLUMNS)])
    b = FeatureBuilder(early, ref).transform(early.pa[list(QUERY_COLUMNS)])
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), atol=1e-10)


def test_shape_and_no_missing(season):
    X = FeatureBuilder(season, "2026-07-01").transform(season.pa[list(QUERY_COLUMNS)])
    assert list(X.columns) == FEATURES
    assert len(FEATURES) == sum(len(v) for v in FEATURE_GROUPS.values()) == len(set(FEATURES))
    assert not X.isna().any().any()


def test_no_history_means_league_rates(season):
    fb = FeatureBuilder(season, "2026-07-01")
    q = season.pa.head(1)[list(QUERY_COLUMNS)].assign(batter=1, pitcher=2)  # unknown players
    X = fb.transform(q)
    assert np.isclose(X["bat_K"].iat[0], fb.league[0])
    assert np.isclose(X["log5_HR"].iat[0], fb.league[5], rtol=1e-6)
    assert X["bvp_pa_log"].iat[0] == 0


def test_log5_is_a_distribution(season):
    X = FeatureBuilder(season, "2026-07-01").transform(season.pa[list(QUERY_COLUMNS)])
    log5 = X[[c for c in FEATURES if c.startswith("log5_")]]
    np.testing.assert_allclose(log5.sum(axis=1), 1.0)
    assert (log5 > 0).all().all()


def test_row_order_is_preserved(season):
    fb = FeatureBuilder(season, "2026-07-01")
    q = season.pa[list(QUERY_COLUMNS)].sample(300, random_state=1)
    X = fb.transform(q)
    assert list(X.index) == list(q.index)
    np.testing.assert_allclose(X.loc[q.index[:50]].to_numpy(), fb.transform(q.iloc[:50]).to_numpy())
