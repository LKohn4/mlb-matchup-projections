"""Single-game report: figures, CSV tables and a Markdown summary."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from matchup import plots
from matchup.evaluate import to_markdown_table
from matchup.project import GameProjection

logger = logging.getLogger(__name__)


def write_game_report(proj: GameProjection, out_dir: Path, credit: str,
                      figure_prefix: str = "") -> dict[str, Path]:  # fmt: skip
    out_dir.mkdir(parents=True, exist_ok=True)
    card, d = proj.card, proj.directory
    away, home = proj.teams[0].team, proj.teams[1].team
    info = (f"{card.date}  ·  at {card.park}  ·  {card.time} game  ·  "
            "projections use 2026 games played before this date")  # fmt: skip
    summary = proj.sim.game_summary()
    lines, pitchers = proj.batter_lines(), proj.pitcher_lines()
    starters = (d.name(proj.teams[0].staff[1]), d.name(proj.teams[1].staff[1]))

    figs = {}
    figs["preview"] = plots.game_preview(summary, away, home, info, starters, proj.sim.score,
                                         out_dir / f"{figure_prefix}game_preview.png", credit)  # fmt: skip
    figs["matchups"] = plots.matchup_heatmaps(
        [proj.matchup_matrix(0), proj.matchup_matrix(1)], [away, home],
        out_dir / f"{figure_prefix}matchup_heatmap.png",
        f"{away} at {home}, {card.date}: projected wOBA per plate appearance for every hitter "
        "against every pitcher he could face (first time facing him).", credit,
    )  # fmt: skip
    rel = [_reliever_matrix(proj.reliever_scenarios(t)) for t in (0, 1)]
    if all(len(m.columns) for m in rel):
        figs["relievers"] = plots.matchup_heatmaps(
            rel, [away, home], out_dir / f"{figure_prefix}reliever_scenarios.png",
            f"If each available reliever comes in: chance of a hit in one plate appearance, "
            f"{away} at {home}, {card.date}.", credit, title="Bullpen Scenarios",
            label="Chance of a hit per PA",
        )  # fmt: skip
    figs["lines"] = plots.projected_lines_table(
        lines, [away, home], out_dir / f"{figure_prefix}projected_lines.png",
        f"{away} at {home}, {card.date}: average of {summary['sims']:,} simulated games. "
        "1+ H = chance of at least one hit. Highlighted: best projected wOBA.", credit,
    )  # fmt: skip

    lines.to_csv(out_dir / "batter_projections.csv", index=False)
    pitchers.to_csv(out_dir / "pitcher_projections.csv", index=False)
    proj.pa_vs_pitchers().to_csv(out_dir / "expected_pa_vs_pitchers.csv", index=False)
    pd.concat([proj.reliever_scenarios(0).assign(team=away),
               proj.reliever_scenarios(1).assign(team=home)]).to_csv(
        out_dir / "reliever_scenarios.csv", index=False)  # fmt: skip
    proj.bench_table().to_csv(out_dir / "bench_projections.csv", index=False)
    for t, team in ((0, away), (1, home)):
        proj.matchup_matrix(t).to_csv(out_dir / f"matchup_matrix_{team}.csv")
    (out_dir / "summary.json").write_text(json.dumps({**summary, "away": away, "home": home,
                                                      "date": card.date, "park": card.park,
                                                      "time": card.time,
                                                      "warnings": proj.warnings}, indent=2),
                                          encoding="utf-8")  # fmt: skip
    (out_dir / "report.md").write_text(_markdown(proj, summary, lines, pitchers, figure_prefix),
                                       encoding="utf-8")  # fmt: skip
    logger.info("Game report written to %s", out_dir)
    return figs


def _reliever_matrix(scen: pd.DataFrame) -> pd.DataFrame:
    if scen.empty:
        return pd.DataFrame()
    scen = scen.assign(col=scen["reliever"] + "\n(" + scen["role"] + ")")
    return scen.pivot(index="batter", columns="col", values="p_hit").reindex(
        index=scen["batter"].drop_duplicates(), columns=scen["col"].drop_duplicates())


def _markdown(proj, summary, lines, pitchers, prefix) -> str:
    card, away, home = proj.card, proj.teams[0].team, proj.teams[1].team
    parts = [f"# {away} at {home} — {card.date}", "",
             f"Park: {card.park} · {card.time} game · {summary['sims']:,} simulations", "",
             f"**Win probability:** {away} {1 - summary['home_win_prob']:.0%} · "
             f"{home} {summary['home_win_prob']:.0%}  ",
             f"**Projected runs:** {away} {summary['away_runs']:.1f} · {home} {summary['home_runs']:.1f}",
             "", f"![Preview]({prefix}game_preview.png)", ""]  # fmt: skip
    cols = {"slot": "#", "name": "Batter", "pa": "PA", "h": "H", "hr": "HR", "k": "K", "bb": "BB",
            "p_1plus_h": "P(1+ H)", "xwoba": "xwOBA"}  # fmt: skip
    for t, team in ((0, away), (1, home)):
        d = lines.loc[lines["team"] == t, list(cols)].rename(columns=cols).round(3)
        parts += [f"## {team} batters", "", to_markdown_table(d, "{:.3f}"), ""]
    pcols = {"team_code": "Team", "role": "Role", "name": "Pitcher", "p_appear": "P(appears)",
             "ip": "IP", "k": "K", "h": "H", "bb": "BB", "r": "R"}  # fmt: skip
    parts += ["## Pitchers", "", to_markdown_table(pitchers[list(pcols)].rename(columns=pcols), "{:.2f}"), ""]
    if proj.warnings:
        parts += ["## Notes", "", *[f"- {w}" for w in proj.warnings], ""]
    parts += [f"![Matchups]({prefix}matchup_heatmap.png)", f"![Bullpen]({prefix}reliever_scenarios.png)",
              f"![Lines]({prefix}projected_lines.png)", ""]  # fmt: skip
    return "\n".join(parts)
