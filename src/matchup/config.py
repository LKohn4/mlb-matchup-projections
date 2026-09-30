"""Constants and run configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

RANDOM_STATE = 42

# --------------------------------------------------------------------------- outcomes

# Every plate appearance is classified into one of these seven outcomes.
OUTCOMES: tuple[str, ...] = ("K", "BB", "1B", "2B", "3B", "HR", "OUT")
K, BB, SINGLE, DOUBLE, TRIPLE, HR, OUT = range(7)
HIT_CODES = (SINGLE, DOUBLE, TRIPLE, HR)
TOTAL_BASES = (0, 0, 1, 2, 3, 4, 0)

EVENT_TO_OUTCOME: dict[str, int] = {
    "strikeout": K,
    "strikeout_double_play": K,
    "walk": BB,
    "hit_by_pitch": BB,
    "single": SINGLE,
    "double": DOUBLE,
    "triple": TRIPLE,
    "home_run": HR,
    "field_out": OUT,
    "force_out": OUT,
    "grounded_into_double_play": OUT,
    "double_play": OUT,
    "fielders_choice": OUT,
    "fielders_choice_out": OUT,
    "field_error": OUT,
    "sac_fly": OUT,
    "sac_fly_double_play": OUT,
    "triple_play": OUT,
}
# Plate appearances that count toward batting order and times-through-the-order but
# aren't modeled (intentional walks and bunts aren't pitcher-vs-batter contests).
UNMODELED_PA_EVENTS = frozenset(
    {"intent_walk", "sac_bunt", "sac_bunt_double_play", "catcher_interf"}
)
OUTS_ON_EVENT: dict[str, int] = {
    "strikeout": 1, "strikeout_double_play": 2, "field_out": 1, "force_out": 1,
    "grounded_into_double_play": 2, "double_play": 2, "fielders_choice_out": 1,
    "sac_fly": 1, "sac_fly_double_play": 2, "triple_play": 3, "sac_bunt": 1,
    "sac_bunt_double_play": 2,
}  # fmt: skip

# Approximate wOBA linear weights, used to summarize a probability distribution as one
# number ("projected wOBA per plate appearance").
WOBA_WEIGHTS = (0.0, 0.70, 0.89, 1.27, 1.61, 2.07, 0.0)

# --------------------------------------------------------------------------- pitches

PITCH_FAMILIES: dict[str, str] = {
    "FF": "fastball", "SI": "fastball", "FC": "fastball", "FA": "fastball",
    "SL": "breaking", "ST": "breaking", "CU": "breaking", "KC": "breaking",
    "SV": "breaking", "CS": "breaking",
    "CH": "offspeed", "FS": "offspeed", "FO": "offspeed", "SC": "offspeed",
    "EP": "offspeed", "KN": "offspeed",
}  # fmt: skip
FAMILIES: tuple[str, ...] = ("fastball", "breaking", "offspeed")
SWING_DESCRIPTIONS = frozenset(
    {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play",
     "foul_bunt", "missed_bunt", "bunt_foul_tip"}
)  # fmt: skip
WHIFF_DESCRIPTIONS = frozenset({"swinging_strike", "swinging_strike_blocked", "missed_bunt"})
HARD_HIT_MPH = 95.0

# --------------------------------------------------------------------------- shrinkage

# How many plate appearances (batters) or batters faced (pitchers) of evidence it takes
# before a player's own rate counts as much as the league average. Rare outcomes need
# more evidence. These are rough reliability constants; the backtest measures the result.
BATTER_SHRINK = {"K": 60, "BB": 120, "1B": 400, "2B": 800, "3B": 1500, "HR": 300, "OUT": 300}
PITCHER_SHRINK = {"K": 70, "BB": 170, "1B": 600, "2B": 1000, "3B": 2000, "HR": 600, "OUT": 400}
SPLIT_SHRINK = 200  # platoon splits shrink toward the player's overall rates
BVP_SHRINK = 40  # batter-vs-pitcher history is heavily shrunk: samples are tiny
RECENT_SHRINK = 20  # hot/cold windows
FAMILY_SWING_SHRINK = 50
FAMILY_PA_SHRINK = 80
USAGE_SHRINK = 100
HARD_HIT_SHRINK = 50
PARK_SHRINK = 1500
HOT_COLD_DAYS = 14

# --------------------------------------------------------------------------- simulator

# In-game context used when projecting each (batter, pitcher) pair at each "stage".
# Starters face the order up to four times; relievers usually face batters on their
# 3rd-5th trip to the plate. Pitch counts are typical values at each stage.
STARTER_STAGE_CONTEXT = {"tto": (1, 2, 3, 4), "pa_num": (1, 2, 3, 4), "pitches": (15, 50, 80, 100)}
RELIEVER_STAGE_CONTEXT = {"tto": (1, 2, 3, 4), "pa_num": (3, 4, 5, 5), "pitches": (10, 35, 60, 80)}
OPENER_STAGE_CONTEXT = {"tto": (1, 2, 3, 4), "pa_num": (1, 2, 3, 4), "pitches": (10, 40, 70, 90)}


def _project_root() -> Path:
    """Folder holding pyproject.toml, so files land in the project wherever it's run from."""
    candidate = Path(__file__).resolve().parents[2]
    return candidate if (candidate / "pyproject.toml").exists() else Path.cwd()


PROJECT_ROOT = _project_root()


@dataclass
class Config:
    """Settings for one run."""

    # 2026 regular season only: Opening Night (Mar 25) through the final day (Sep 27).
    start_date: str = "2026-03-25"
    end_date: str = "2026-09-27"
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    results_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "results")
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models")
    cards_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "game_cards")
    refresh_data: bool = False
    fetch_external: bool = True  # day/night schedule and player names from MLB Stats API
    sample_frac: float | None = None  # fraction of *games* kept, for quick runs

    # Walk-forward backtest: train on everything before each cutoff, test until the next.
    fold_cutoffs: tuple[str, ...] = ("2026-07-01", "2026-08-01", "2026-09-01")

    n_estimators: int = 150
    rf_min_samples_leaf: int = 80
    hgb_max_iter: int = 500
    run_nn: bool = True
    nn_epochs: int = 30
    nn_batch_size: int = 256

    run_ablation: bool = True
    run_game_backtest: bool = True
    backtest_games: int = 150
    backtest_sims: int = 1000

    sims: int = 10_000  # simulations per projected game
    credit: str = "Leo Kohn  |  github.com/LKohn4"

    @property
    def cache_path(self) -> Path:
        # Same file name as the mlb-hit-probability project, so its download can be reused.
        return self.data_dir / f"statcast_{self.start_date}_{self.end_date}.pkl"

    @property
    def schedule_path(self) -> Path:
        return self.data_dir / f"schedule_{self.start_date}_{self.end_date}.csv"

    @property
    def players_path(self) -> Path:
        return self.data_dir / f"players_{self.start_date[:4]}.csv"

    @property
    def figures_dir(self) -> Path:
        return self.results_dir / "figures"

    @property
    def season_label(self) -> str:
        return f"{self.start_date[:4]} MLB regular season"
