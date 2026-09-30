import numpy as np
import pytest

from matchup.config import Config
from matchup.features import QUERY_COLUMNS, FeatureBuilder
from matchup.models import (
    GradientBoosting,
    LeagueAverage,
    Log5,
    RandomForest,
    load_model,
    make_models,
    save_model,
)


@pytest.fixture(scope="module")
def xy(season):
    X = FeatureBuilder(season, "2026-07-01").transform(season.pa[list(QUERY_COLUMNS)])
    return X, season.pa["outcome"].to_numpy()


@pytest.mark.parametrize("model", [LeagueAverage(), Log5(), RandomForest(20), GradientBoosting(40)],
                         ids=lambda m: m.name)
def test_models_return_valid_distributions(xy, model):
    X, y = xy
    P = model.fit(X, y).predict_proba(X.iloc[:500])
    assert P.shape == (500, 7)
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-6)
    assert (P > 0).all()


def test_missing_class_gets_a_column(xy):
    X, y = xy
    keep = y != 4  # no triples in training
    P = RandomForest(10).fit(X[keep], y[keep]).predict_proba(X.iloc[:10])
    assert P.shape == (10, 7)


def test_save_and_load_roundtrip(xy, tmp_path):
    X, y = xy
    model = GradientBoosting(20).fit(X, y)
    save_model(model, tmp_path, {"model": model.name})
    loaded, meta = load_model(tmp_path)
    assert meta["model"] == "Gradient boosting"
    np.testing.assert_allclose(loaded.predict_proba(X.iloc[:20]), model.predict_proba(X.iloc[:20]))


def test_load_model_explains_missing_training(tmp_path):
    with pytest.raises(FileNotFoundError, match="Train models"):
        load_model(tmp_path)


def test_make_models_respects_nn_switch():
    names = [m.name for m in make_models(Config(run_nn=False))]
    assert names == ["League average", "Log5", "Random forest", "Gradient boosting"]


def test_neural_network(xy):
    pytest.importorskip("tensorflow")
    from matchup.models import NeuralNetwork

    X, y = xy
    P = NeuralNetwork(epochs=2).fit(X.iloc[:3000], y[:3000]).predict_proba(X.iloc[:100])
    assert P.shape == (100, 7)
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-5)
