"""End-to-end workflows: train and backtest, make a game card, project a game."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from matchup import __version__
from matchup.backtest import ablation, game_backtest, game_calibration, walk_forward
from matchup.config import PROJECT_ROOT, Config
from matchup.dashboard import build_dashboard, insight_tables
from matchup.data import (
    SeasonData,
    actual_team_runs,
    coverage_report,
    load_statcast,
    prepare,
    sample_games,
    validate_columns,
)
from matchup.evaluate import to_markdown_table
from matchup.external import day_night_map, load_player_names, load_schedule
from matchup.features import QUERY_COLUMNS, FeatureBuilder
from matchup.gamecard import generate_card, load_card, save_card
from matchup.models import (
    GradientBoosting,
    LeagueAverage,
    Log5,
    NeuralNetwork,
    RandomForest,
    load_model,
    save_model,
)
from matchup.players import PlayerDirectory
from matchup.project import GameProjection, project_game
from matchup.report import write_game_report
from matchup.simulate import StaffPlan, simulate_games

logger = logging.getLogger(__name__)


@dataclass
class Loaded:
    raw: pd.DataFrame
    season: SeasonData
    directory: PlayerDirectory
    names: pd.DataFrame
    coverage: dict


def load_season(config: Config) -> Loaded:
    raw = load_statcast(config.start_date, config.end_date, config.cache_path, config.refresh_data)
    validate_columns(raw)
    coverage = coverage_report(raw, config.end_date)
    if config.sample_frac:
        raw = sample_games(raw, config.sample_frac)
        logger.info("Sampled %.0f%% of games for a quick run", 100 * config.sample_frac)
    schedule = load_schedule(config.start_date, config.end_date, config.schedule_path,
                             config.fetch_external)  # fmt: skip
    season = prepare(raw, day_night_map(schedule) if len(schedule) else None)
    ids = pd.concat([season.pa_all["batter"], season.pa_all["pitcher"]]).unique()
    names = load_player_names(ids, config.players_path, season.pitches, config.fetch_external)
    directory = PlayerDirectory.build(season.pa_all, names)
    return Loaded(raw, season, directory, names, coverage)


def _model_by_name(name: str, config: Config):
    return {"League average": LeagueAverage, "Log5": Log5,
            "Random forest": lambda: RandomForest(config.n_estimators, config.rf_min_samples_leaf),
            "Gradient boosting": lambda: GradientBoosting(config.hgb_max_iter),
            "Neural network": lambda: NeuralNetwork(config.nn_epochs, config.nn_batch_size)}[name]()  # fmt: skip


def _sim_check(season: SeasonData, raw: pd.DataFrame) -> dict:
    """Does a league-average game in the simulator score like a real 2026 game?"""
    league = np.bincount(season.pa["outcome"], minlength=7) / len(season.pa)
    sim = simulate_games(np.broadcast_to(league, (2, 9, 6, 4, 7)).copy(), (StaffPlan(), StaffPlan()),
                         n_sims=20_000)  # fmt: skip
    actual = actual_team_runs(raw)
    return {"sim_runs_per_team": sim.game_summary()["total_runs"] / 2,
            "actual_runs_per_team": float(actual.mean()) if actual is not None else None}  # fmt: skip


def _showcase_teams(raw: pd.DataFrame, directory: PlayerDirectory) -> tuple[str, str]:
    """The two best teams by run differential (for the example game)."""
    runs = actual_team_runs(raw)
    if runs is None:
        teams = sorted(directory.teams)
        return teams[0], teams[1]
    df = runs.rename("runs").reset_index()
    against = df.merge(df, on="game_pk", suffixes=("", "_opp"))
    against = against.loc[against["bat_team"] != against["bat_team_opp"]]
    diff = (against["runs"] - against["runs_opp"]).groupby(against["bat_team"]).sum()
    best = diff.sort_values(ascending=False).index[:2].tolist()
    return best[1], best[0]  # best team hosts


def train(config: Config) -> dict:
    loaded = load_season(config)
    season, directory = loaded.season, loaded.directory

    logger.info("Walk-forward backtest (%s)", ", ".join(config.fold_cutoffs))
    bt = walk_forward(season, config)
    if config.run_ablation:
        logger.info("Ablation: retraining without each variable group")
        bt.ablation = ablation(bt, config)
    if config.run_game_backtest:
        dir_cut = PlayerDirectory.build(season.pa_all, loaded.names, as_of=bt.last_fold["cutoff"])
        bt.game_rows = game_backtest(season, bt, directory, dir_cut, config)
        if len(bt.game_rows):
            bt.game_calibration = game_calibration(bt.game_rows)

    # Final model: the backtest winner, retrained on the whole season.
    reference_end = season.last_date + pd.Timedelta(days=1)
    builder = FeatureBuilder(season, reference_end)
    X_all = builder.transform(season.pa[list(QUERY_COLUMNS)])
    y_all = season.pa["outcome"].to_numpy()
    final = _model_by_name(bt.best_model, config)
    logger.info("Training final %s on all %s plate appearances", bt.best_model, f"{len(y_all):,}")
    final.fit(X_all, y_all)
    metadata = {
        "model": bt.best_model, "version": __version__, "season": config.season_label,
        "start_date": config.start_date, "end_date": config.end_date,
        "trained_through": str(season.last_date.date()), "plate_appearances": int(len(y_all)),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backtest": bt.pooled_scores.round(5).to_dict(orient="records"),
    }  # fmt: skip
    save_model(final, config.models_dir, metadata)

    # Dashboard.
    check = _sim_check(season, loaded.raw)
    best = bt.pooled_scores.set_index("model").loc[bt.best_model]
    kpis = [
        (f"{len(season.pa):,}", "Plate appearances", "modeled, 2026 regular season"),
        (f"{season.pa_all['game_pk'].nunique():,}", "Games", f"through {season.last_date.date()}"),
        (f"{best['skill_vs_league_pct']:+.2f}%", "Log-loss skill", f"{bt.best_model} vs. league average"),
        (f"{check['sim_runs_per_team']:.2f}", "Simulated runs/team",
         f"actual 2026: {check['actual_runs_per_team']:.2f}" if check["actual_runs_per_team"]
         else "league-average matchup"),
    ]  # fmt: skip
    tables = insight_tables(season, builder, X_all)
    figures = build_dashboard(tables, bt, kpis, config)

    # Example game between the season's two best teams.
    away, home = _showcase_teams(loaded.raw, directory)
    date = str((season.last_date + pd.Timedelta(days=3)).date())
    card = generate_card(season.pa_all, directory, away, home, date, "night", config.sims)
    card_path = save_card(card, config.cards_dir / f"example_{card.slug}.yaml",
                          note="Auto-generated example: edit any name and re-run 'project'.")  # fmt: skip
    proj = project_game(card, directory, builder, final, sims=config.sims)
    game_dir = config.results_dir / "games" / card.slug
    write_game_report(proj, game_dir, config.credit)
    for png in game_dir.glob("*.png"):
        shutil.copy(png, config.figures_dir / f"example_{png.name}")
        figures.append(config.figures_dir / f"example_{png.name}")

    # Written results.
    results = config.results_dir
    bt.fold_scores.to_csv(results / "fold_scores.csv", index=False)
    bt.pooled_scores.to_csv(results / "model_scores.csv", index=False)
    if bt.game_rows is not None:
        bt.game_rows.to_csv(results / "game_backtest.csv", index=False)
    cols = {"model": "Model", "log_loss": "Log loss", "brier": "Brier",
            "skill_vs_league_pct": "Skill vs league %", "skill_vs_log5_pct": "Skill vs log5 %"}  # fmt: skip
    table = bt.pooled_scores[list(cols)].rename(columns=cols)
    (results / "model_scorecard.md").write_text(
        f"# Model scorecard ({config.season_label})\n\nWalk-forward backtest, "
        f"{len(bt.fold_scores['fold'].unique())} folds, {int(bt.pooled_scores['n'].iloc[0]):,} "
        f"test plate appearances. Best: **{bt.best_model}**.\n\n{to_markdown_table(table)}\n",
        encoding="utf-8")  # fmt: skip
    summary = {**metadata, "coverage": loaded.coverage, "sim_check": check,
               "example_card": str(card_path.relative_to(PROJECT_ROOT)) if card_path.is_relative_to(PROJECT_ROOT) else str(card_path),
               "figures": [p.name for p in figures],
               "ablation": bt.ablation.round(5).to_dict(orient="records") if bt.ablation is not None else None}  # fmt: skip
    (results / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("Done: %d figures in %s", len(figures), config.figures_dir)
    return summary


def make_card(config: Config, away: str, home: str, date: str | None, time: str,
              out: Path | None) -> Path:  # fmt: skip
    loaded = load_season(config)
    date = date or str((loaded.season.last_date + pd.Timedelta(days=1)).date())
    card = generate_card(loaded.season.pa_all, loaded.directory, away, home, date, time, config.sims)
    out = out or config.cards_dir / f"{card.slug}.yaml"
    return save_card(card, out, note="Auto-filled from 2026 usage; edit anything before projecting.")


def project(config: Config, card_path: Path, sims: int | None = None,
            out_dir: Path | None = None) -> GameProjection:  # fmt: skip
    card = load_card(card_path)
    model, meta = load_model(config.models_dir)
    logger.info("Using %s trained through %s", meta["model"], meta["trained_through"])
    loaded = load_season(config)
    season = loaded.season
    reference_end = min(pd.Timestamp(card.date), season.last_date + pd.Timedelta(days=1))
    builder = FeatureBuilder(season, reference_end)
    proj = project_game(card, loaded.directory, builder, model, sims=sims)
    out_dir = out_dir or config.results_dir / "games" / card.slug
    write_game_report(proj, out_dir, config.credit)
    return proj


def search_players(config: Config, query: str) -> pd.DataFrame:
    return load_season(config).directory.search(query)
