"""End-to-end smoke test on synthetic data with fast settings."""

import json

import pytest
from synthetic import fake_names, make_statcast

from matchup import pipeline
from matchup.config import Config


@pytest.fixture(scope="module")
def config(tmp_path_factory):
    root = tmp_path_factory.mktemp("run")
    cfg = Config(data_dir=root / "data", results_dir=root / "results", models_dir=root / "models",
                 cards_dir=root / "cards", fetch_external=False, run_nn=False, n_estimators=20,
                 fold_cutoffs=("2026-07-15", "2026-08-20"), backtest_games=8, backtest_sims=100,
                 sims=300, credit="test")
    raw = make_statcast(n_games=260, seed=11)
    cfg.data_dir.mkdir(parents=True)
    raw.to_pickle(cfg.cache_path)
    fake_names(raw).to_csv(cfg.players_path, index=False)
    return cfg


def test_train_card_and_project(config):
    summary = pipeline.train(config)
    figures = sorted(p.name for p in config.figures_dir.glob("*.png"))
    assert len(figures) >= 14, figures
    assert (config.models_dir / "metadata.json").exists()
    saved = json.loads((config.results_dir / "summary.json").read_text())
    assert saved["model"] == summary["model"]

    card = pipeline.make_card(config, "BOS", "HOU", "2026-10-01", "day", None)
    proj = pipeline.project(config, card, sims=200)
    lines = proj.batter_lines()
    assert len(lines) == 18 and lines["pa"].between(3, 6).all()
    out = config.results_dir / "games" / proj.card.slug
    assert (out / "report.md").exists() and (out / "batter_projections.csv").exists()
