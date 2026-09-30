"""Season and model dashboard: figures plus the CSV data behind each one."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from matchup import plots
from matchup.config import FAMILIES, Config, K
from matchup.data import SeasonData

logger = logging.getLogger(__name__)


def insight_tables(season: SeasonData, builder, X_all: pd.DataFrame) -> dict[str, pd.DataFrame]:
    pa, pitches = season.pa, season.pitches
    tto = pa.groupby("tto").agg(k_rate=("outcome", lambda s: (s == K).mean()), woba=("woba", "mean"),
                                n=("outcome", "size")).reset_index()  # fmt: skip
    platoon = pa.groupby(["stand", "p_throws"]).agg(woba=("woba", "mean"), n=("woba", "size")).reset_index()

    fam_p = pitches.dropna(subset=["family"])
    whiff = fam_p.groupby(["family", "stand"]).agg(sw=("is_swing", "sum"), wh=("is_whiff", "sum"))
    woba = pa.dropna(subset=["family"]).groupby(["family", "stand"])["woba"].mean()
    family = whiff.join(woba).reset_index()
    family["whiff_rate"] = family["wh"] / family["sw"].clip(lower=1)
    family["family"] = pd.Categorical(family["family"], FAMILIES, ordered=True)
    family = family.sort_values(["family", "stand"])
    family["family"] = family["family"].astype(str)

    # Hot/cold: recent form vs. what happened next, for batters with a real recent sample.
    ob_cols = ["bat_BB", "bat_1B", "bat_2B", "bat_3B", "bat_HR"]
    expected_ob = X_all[ob_cols].sum(axis=1)
    hc = pd.DataFrame({"recent_delta": X_all["bat_recent_ob_delta"],
                       "next_delta": pa["on_base"].to_numpy() - expected_ob.to_numpy(),
                       "recent_pa": np.expm1(X_all["bat_recent_pa_log"])})  # fmt: skip
    hc = hc.loc[hc["recent_pa"] >= 20]
    if len(hc) > 200:
        hc["bin"] = pd.qcut(hc["recent_delta"], 10, labels=False, duplicates="drop")
        hot_cold = hc.groupby("bin").agg(recent_delta=("recent_delta", "mean"),
                                         next_delta=("next_delta", "mean"),
                                         n=("next_delta", "size")).reset_index(drop=True)  # fmt: skip
    else:
        hot_cold = pd.DataFrame(columns=["recent_delta", "next_delta", "n"])

    parks = pd.DataFrame({"park": list(builder.park_hr),
                          "hr_factor": list(builder.park_hr.values()),
                          "hit_factor": [builder.park_hit[p] for p in builder.park_hr]})  # fmt: skip

    rows = []
    for value, label in ((1, "Home batters"), (0, "Away batters")):
        d = pa.loc[pa["is_home"] == value, "woba"]
        rows.append({"kind": "venue", "label": label, "woba": d.mean(), "n": len(d)})
    for value, label in ((0.0, "Day games"), (1.0, "Night games")):
        d = pa.loc[pa["is_night"] == value, "woba"]
        if len(d):
            rows.append({"kind": "time", "label": label, "woba": d.mean(), "n": len(d)})
    situational = pd.DataFrame(rows)
    return {"times_through_order": tto, "platoon_splits": platoon, "pitch_families": family,
            "hot_cold": hot_cold, "park_factors": parks, "situational": situational}  # fmt: skip


def build_dashboard(tables: dict, backtest, kpis, config: Config) -> list[Path]:
    fig_dir, data_dir = config.figures_dir, config.results_dir / "dashboard_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    season, credit = config.season_label, config.credit
    for name, df in tables.items():
        df.to_csv(data_dir / f"{name}.csv", index=False)
    made = [
        plots.kpi_card(kpis, fig_dir / "season_summary.png", "Matchup Projection Model", season, credit),
        plots.model_scorecard(backtest.pooled_scores, fig_dir / "model_scorecard.png", season, credit),
        plots.calibration_by_outcome(backtest.calibration, fig_dir / "calibration_by_outcome.png",
                                     backtest.best_model, season, credit),  # fmt: skip
        plots.tto_chart(tables["times_through_order"], fig_dir / "times_through_order.png", season, credit),
        plots.platoon_chart(tables["platoon_splits"], fig_dir / "platoon_splits.png", season, credit),
        plots.pitch_family_chart(tables["pitch_families"], fig_dir / "pitch_families.png", season, credit),
        plots.park_factors_chart(tables["park_factors"], fig_dir / "park_factors.png", season, credit),
        plots.situational_chart(tables["situational"], fig_dir / "home_away_day_night.png", season, credit),
    ]
    if len(tables["hot_cold"]):
        made.append(plots.hot_cold_chart(tables["hot_cold"], fig_dir / "hot_cold.png", season, credit))
    if backtest.ablation is not None and len(backtest.ablation):
        backtest.ablation.to_csv(data_dir / "ablation.csv", index=False)
        made.append(plots.ablation_chart(backtest.ablation, fig_dir / "variable_importance.png",
                                         season, credit))  # fmt: skip
    if backtest.game_calibration is not None and len(backtest.game_calibration):
        backtest.game_calibration.to_csv(data_dir / "game_calibration.csv", index=False)
        n_games = int(backtest.game_rows["game_pk"].nunique())
        made.append(plots.game_calibration_chart(backtest.game_calibration,
                                                 fig_dir / "game_backtest.png", n_games, season,
                                                 credit))  # fmt: skip
    backtest.calibration.to_csv(data_dir / "calibration_by_outcome.csv", index=False)
    logger.info("Dashboard: %d figures in %s", len(made), fig_dir)
    return made
