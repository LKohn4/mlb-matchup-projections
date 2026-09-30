"""Walk-forward backtesting: score projections the way they'd actually be used.

For each cutoff date, models train only on plate appearances before the cutoff and are
scored on the weeks after it. Features always come from games before each plate
appearance, so nothing from the future leaks in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from matchup.config import HIT_CODES, OUTCOMES, Config, K
from matchup.data import SeasonData
from matchup.evaluate import binary_calibration, brier, calibration_table, log_loss, score_model
from matchup.features import FEATURE_GROUPS, FEATURES, QUERY_COLUMNS, FeatureBuilder
from matchup.gamecard import card_from_game
from matchup.models import GradientBoosting, fit_timed, make_models
from matchup.players import PlayerDirectory
from matchup.project import assemble, build_queries, resolve_card
from matchup.simulate import simulate_games

logger = logging.getLogger(__name__)
BASELINES = ("League average", "Log5")


@dataclass
class BacktestResult:
    fold_scores: pd.DataFrame
    pooled_scores: pd.DataFrame
    calibration: pd.DataFrame
    best_model: str
    last_fold: dict = field(default_factory=dict)
    ablation: pd.DataFrame | None = None
    game_rows: pd.DataFrame | None = None
    game_calibration: pd.DataFrame | None = None


def walk_forward(season: SeasonData, config: Config) -> BacktestResult:
    cutoffs = [pd.Timestamp(c) for c in config.fold_cutoffs]
    end = season.last_date + pd.Timedelta(days=1)
    bounds = [*cutoffs, end]
    fold_rows, pooled_P, pooled_y = [], {}, []
    last_fold: dict = {}

    for i, cutoff in enumerate(cutoffs):
        stop = bounds[i + 1]
        rows = season.pa.loc[season.pa["game_date"] < stop]
        train = (rows["game_date"] < cutoff).to_numpy()
        test = ~train
        label = f"{cutoff.date()} to {(stop - pd.Timedelta(days=1)).date()}"
        if train.sum() < 500 or test.sum() < 100:
            logger.warning("Skipping fold %s: not enough data (%d train, %d test)", label,
                           train.sum(), test.sum())  # fmt: skip
            continue
        logger.info("Fold %s: training on %s PAs, testing on %s", label,
                    f"{train.sum():,}", f"{test.sum():,}")  # fmt: skip
        builder = FeatureBuilder(season, cutoff)
        X = builder.transform(rows[list(QUERY_COLUMNS)])
        y = rows["outcome"].to_numpy()
        models = make_models(config)
        baseline_ll = None
        fold_models = {}
        for model in models:
            t = fit_timed(model, X[train], y[train])
            P = model.predict_proba(X[test])
            row = score_model(model.name, label, y[test], P, t, baseline_ll)
            if baseline_ll is None:
                baseline_ll = row["log_loss"]
                row["skill_vs_league_pct"] = 0.0
            fold_rows.append(row)
            pooled_P.setdefault(model.name, []).append(P)
            fold_models[model.name] = model
            logger.info("  %-18s log loss %.4f  (%+.2f%% vs league)", model.name,
                        row["log_loss"], row["skill_vs_league_pct"])  # fmt: skip
        pooled_y.append(y[test])
        last_fold = {"cutoff": cutoff, "stop": stop, "builder": builder, "models": fold_models,
                     "X": X, "y": y, "train": train, "test": test}  # fmt: skip

    if not fold_rows:
        raise RuntimeError("No backtest fold had enough data. Check the fold cutoff dates.")
    y_all = np.concatenate(pooled_y)
    pooled = {name: np.vstack(parts) for name, parts in pooled_P.items()}
    league_ll = log_loss(y_all, pooled["League average"])
    league_brier = brier(y_all, pooled["League average"])
    pooled_rows = []
    for name, P in pooled.items():
        row = score_model(name, "All folds", y_all, P, 0.0, league_ll)
        row["brier_skill_pct"] = 100 * (league_brier - row["brier"]) / league_brier
        row["skill_vs_log5_pct"] = 100 * (log_loss(y_all, pooled["Log5"]) - row["log_loss"]) / league_ll
        pooled_rows.append(row)
    pooled_scores = pd.DataFrame(pooled_rows).drop(columns=["train_time_s", "fold"])
    candidates = pooled_scores.loc[pooled_scores["model"] != "League average"]
    best = str(candidates.loc[candidates["log_loss"].idxmin(), "model"])
    logger.info("Best model across folds: %s", best)
    return BacktestResult(
        fold_scores=pd.DataFrame(fold_rows), pooled_scores=pooled_scores,
        calibration=calibration_table(y_all, pooled[best]), best_model=best, last_fold=last_fold,
    )  # fmt: skip


def ablation(result: BacktestResult, config: Config) -> pd.DataFrame:
    """Retrain gradient boosting without each variable group; how much worse does it get?"""
    f = result.last_fold
    X, y, train, test = f["X"], f["y"], f["train"], f["test"]
    full = GradientBoosting(config.hgb_max_iter)
    full.fit(X[train], y[train])
    base_ll = log_loss(y[test], full.predict_proba(X[test]))
    league = np.bincount(y[train], minlength=len(OUTCOMES)) / train.sum()
    league_ll = log_loss(y[test], np.tile(league, (test.sum(), 1)))
    rows = []
    for group, cols in FEATURE_GROUPS.items():
        keep = [c for c in FEATURES if c not in cols]
        model = GradientBoosting(config.hgb_max_iter, features=keep)
        model.fit(X[train], y[train])
        ll = log_loss(y[test], model.predict_proba(X[test]))
        rows.append({"group": group, "n_features": len(cols), "log_loss_without": ll,
                     "loss_increase_pct": 100 * (ll - base_ll) / league_ll})  # fmt: skip
        logger.info("  Ablation %-26s %+.3f%%", group, rows[-1]["loss_increase_pct"])
    return pd.DataFrame(rows).sort_values("loss_increase_pct", ascending=False)


def game_backtest(season: SeasonData, result: BacktestResult, directory: PlayerDirectory,
                  directory_at_cutoff: PlayerDirectory, config: Config) -> pd.DataFrame:  # fmt: skip
    """Simulate real games from the last fold and compare with what batters actually did."""
    f = result.last_fold
    model, builder = f["models"][result.best_model], f["builder"]
    pa_all = season.pa_all
    window = pa_all.loc[(pa_all["game_date"] >= f["cutoff"]) & (pa_all["game_date"] < f["stop"])]
    games = window["game_pk"].drop_duplicates()
    games = games.sample(n=min(config.backtest_games, len(games)), random_state=42).tolist()
    night = window.groupby("game_pk")["is_night"].first()

    prepared, queries = [], []
    for g in games:
        try:
            card = card_from_game(pa_all, directory, g, float(night.get(g, 0.5)))
            teams, _ = resolve_card(card, directory)
        except Exception as exc:  # unusual games (suspended, missing data) are skipped
            logger.debug("Skipping game %s: %s", g, exc)
            continue
        if any(len(t.lineup) != 9 for t in teams):
            continue
        for t in teams:  # starter workload as known at the cutoff
            t.plan.starter_outs_mean = directory_at_cutoff.avg_start_outs(t.staff[1])
        prepared.append((g, card, teams))
        queries.append(build_queries(card, teams, directory, key=g))
    if not prepared:
        return pd.DataFrame()
    q = pd.concat(queries, ignore_index=True)
    P = model.predict_proba(builder.transform(q))
    logger.info("Game backtest: simulating %d games", len(prepared))

    actual = season.pa.loc[season.pa["game_pk"].isin([g for g, _, _ in prepared])]
    hits = actual.assign(h=actual["outcome"].isin(HIT_CODES), k=actual["outcome"].eq(K))
    per_batter = hits.groupby(["game_pk", "batter"]).agg(h=("h", "sum"), k=("k", "sum"),
                                                         pa=("outcome", "size"))  # fmt: skip
    rows = []
    for g, _card, teams in prepared:
        m = (q["key"] == g).to_numpy()
        grid, _ = assemble(q.loc[m], P[m], teams)
        sim = simulate_games(grid, (teams[0].plan, teams[1].plan), n_sims=config.backtest_sims,
                             seed=int(g) % 10_000)  # fmt: skip
        table = sim.batter_table()
        for r in table.itertuples():
            b = teams[r.team].lineup[r.slot - 1]
            act = per_batter.loc[(g, b)] if (g, b) in per_batter.index else None
            if act is None:
                continue
            rows.append({"game_pk": g, "batter": b, "pred_p1h": r.p_1plus_h, "pred_h": r.h,
                         "pred_p1k": r.p_1plus_k, "pred_pa": r.pa, "actual_h": int(act["h"]),
                         "actual_k": int(act["k"]), "actual_pa": int(act["pa"])})  # fmt: skip
    return pd.DataFrame(rows)


def game_calibration(rows: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for label, pred, actual in (("1+ hit", "pred_p1h", rows["actual_h"] >= 1),
                                ("1+ strikeout", "pred_p1k", rows["actual_k"] >= 1)):  # fmt: skip
        cal = binary_calibration(rows[pred].to_numpy(), actual.to_numpy().astype(float), bins=8)
        parts.append(cal.assign(event=label))
    return pd.concat(parts, ignore_index=True)
