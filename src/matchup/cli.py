"""Command line: ``mlb-matchup train | card | project | players``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from matchup import __version__
from matchup.config import PROJECT_ROOT, Config
from matchup.gamecard import CardError

logger = logging.getLogger("matchup")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mlb-matchup", description=(
        "Project how every player performs in a specific MLB matchup, from 2026 Statcast data."))
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--offline", action="store_true",
                   help="don't call the MLB Stats API (day/night and names use cached files only)")
    p.add_argument("--credit", help="credit line printed on every figure")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="backtest, train the final model, build the dashboard")
    t.add_argument("--quick", action="store_true", help="fast settings for a first test run")
    t.add_argument("--skip-nn", action="store_true", help="skip the neural network")
    t.add_argument("--no-ablation", action="store_true", help="skip the variable-importance study")
    t.add_argument("--no-game-backtest", action="store_true", help="skip simulating real games")
    t.add_argument("--sample-frac", type=float, help="use this fraction of games (e.g. 0.3)")
    t.add_argument("--refresh-data", action="store_true", help="re-download Statcast data")

    c = sub.add_parser("card", help="create a filled-in game card for two teams")
    c.add_argument("away", help="away team code, e.g. LAD")
    c.add_argument("home", help="home team code, e.g. NYY")
    c.add_argument("--date", help="game date (default: day after the last 2026 game)")
    c.add_argument("--time", default="night", choices=["day", "night", "unknown"])
    c.add_argument("--out", type=Path, help="where to save the card")

    j = sub.add_parser("project", help="project a game from a game card")
    j.add_argument("card", type=Path, help="path to a game card .yaml file")
    j.add_argument("--sims", type=int, help="number of simulated games (default: card or 10000)")
    j.add_argument("--out-dir", type=Path, help="report folder (default: results/games/<game>)")

    s = sub.add_parser("players", help="search 2026 players by name")
    s.add_argument("query", help="part of a name, e.g. 'smith'")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")  # fmt: skip
    for noisy in ("matplotlib", "PIL", "urllib3", "absl", "h5py", "tensorflow"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    config = Config()
    if args.offline:
        config.fetch_external = False
    if args.credit is not None:
        config.credit = args.credit
    logger.info("mlb-matchup-projections v%s | %s (%s to %s)", __version__, config.season_label,
                config.start_date, config.end_date)  # fmt: skip
    logger.info("Code: %s | Data: %s | Results: %s", Path(__file__).parent, config.data_dir,
                config.results_dir)  # fmt: skip

    from matchup import pipeline

    try:
        if args.command == "train":
            if args.quick:
                config.n_estimators, config.run_nn, config.run_ablation = 60, False, False
                config.backtest_games, config.backtest_sims, config.sims = 40, 300, 2000
                config.fold_cutoffs = config.fold_cutoffs[-1:]
            config.run_nn = config.run_nn and not args.skip_nn
            config.run_ablation = config.run_ablation and not args.no_ablation
            config.run_game_backtest = not args.no_game_backtest
            config.sample_frac = args.sample_frac
            config.refresh_data = args.refresh_data
            summary = pipeline.train(config)
            logger.info("Best model: %s. Example card: %s", summary["model"], summary["example_card"])
        elif args.command == "card":
            path = pipeline.make_card(config, args.away, args.home, args.date, args.time, args.out)
            logger.info("Game card saved: %s", path)
            logger.info("Edit it, then run:  mlb-matchup project %s",
                        path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path)  # fmt: skip
        elif args.command == "project":
            proj = pipeline.project(config, args.card, args.sims, args.out_dir)
            s = proj.sim.game_summary()
            away, home = proj.teams[0].team, proj.teams[1].team
            print(f"\n{away} {s['away_runs']:.1f} - {home} {s['home_runs']:.1f} runs | "
                  f"{home} win probability {s['home_win_prob']:.1%}\n")
            cols = ["team_code", "slot", "name", "pa", "h", "hr", "k", "p_1plus_h", "xwoba"]
            print(proj.batter_lines()[cols].round(3).to_string(index=False))
        elif args.command == "players":
            print(pipeline.search_players(config, args.query).to_string(index=False))
    except CardError as exc:
        logger.error("%s", exc)
        return 2
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
