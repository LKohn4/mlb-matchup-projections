import pandas as pd
import pytest
from synthetic import fake_names, make_statcast

from matchup.data import prepare
from matchup.players import PlayerDirectory


@pytest.fixture(scope="session")
def raw() -> pd.DataFrame:
    return make_statcast(n_games=220, seed=5)


@pytest.fixture(scope="session")
def season(raw):
    return prepare(raw)


@pytest.fixture(scope="session")
def names(raw):
    return fake_names(raw)


@pytest.fixture(scope="session")
def directory(season, names):
    return PlayerDirectory.build(season.pa_all, names)
