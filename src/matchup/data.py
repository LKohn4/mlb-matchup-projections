"""Load 2026 Statcast pitches and build plate-appearance and pitch tables."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from matchup.config import (
    EVENT_TO_OUTCOME,
    HARD_HIT_MPH,
    HIT_CODES,
    OUT,
    OUTS_ON_EVENT,
    PITCH_FAMILIES,
    SWING_DESCRIPTIONS,
    TOTAL_BASES,
    UNMODELED_PA_EVENTS,
    WHIFF_DESCRIPTIONS,
    WOBA_WEIGHTS,
)

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"game_date", "game_pk", "game_type", "at_bat_number", "pitch_number", "inning",
     "inning_topbot", "home_team", "away_team", "batter", "pitcher", "stand", "p_throws",
     "events", "description", "pitch_type", "launch_speed"}
)  # fmt: skip
OPTIONAL_COLUMNS = ("player_name", "post_bat_score")


@dataclass
class SeasonData:
    pitches: pd.DataFrame  # one row per pitch (regular season)
    pa_all: pd.DataFrame  # one row per plate appearance, including unmodeled ones
    pa: pd.DataFrame  # modeled plate appearances only

    @property
    def last_date(self) -> pd.Timestamp:
        return self.pa_all["game_date"].max()


def load_statcast(start: str, end: str, cache_path: Path, refresh: bool = False) -> pd.DataFrame:
    """Every pitch between two dates, from a local cache when available."""
    if cache_path.exists() and not refresh:
        logger.info("Loading cached Statcast data from %s", cache_path)
        return pd.read_pickle(cache_path)

    from pybaseball import cache, statcast

    cache.enable()
    if refresh and hasattr(cache, "purge"):
        cache.purge()
    logger.info("Downloading Statcast data %s to %s (this can take 10-30 minutes)", start, end)
    pitches = statcast(start_dt=start, end_dt=end)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pitches.to_pickle(cache_path)
    logger.info("Saved %s pitches to %s", f"{len(pitches):,}", cache_path)
    return pitches


def validate_columns(raw: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(raw.columns))
    if missing:
        raise ValueError(
            f"Statcast data is missing required columns: {missing}. Baseball Savant may have "
            "renamed them; update REQUIRED_COLUMNS in data.py to match."
        )


def coverage_report(raw: pd.DataFrame, expected_end: str) -> dict[str, object]:
    regular = raw.loc[raw["game_type"].eq("R")]
    dates = pd.to_datetime(regular["game_date"])
    report = {
        "first_game_date": str(dates.min().date()) if len(dates) else None,
        "last_game_date": str(dates.max().date()) if len(dates) else None,
        "regular_season_games": int(regular["game_pk"].nunique()),
        "pitches": int(len(regular)),
    }
    logger.info("Coverage: %s to %s, %s games, %s pitches", report["first_game_date"],
                report["last_game_date"], f"{report['regular_season_games']:,}",
                f"{report['pitches']:,}")  # fmt: skip
    if report["last_game_date"] and report["last_game_date"] < expected_end:
        logger.warning("Data ends on %s, before %s. Statcast may not have posted the final "
                       "games yet; re-run later with --refresh-data.",
                       report["last_game_date"], expected_end)  # fmt: skip
    return report


def sample_games(raw: pd.DataFrame, frac: float, seed: int = 42) -> pd.DataFrame:
    """Keep a random fraction of whole games (sampling pitches would break game structure)."""
    games = raw["game_pk"].drop_duplicates().sample(frac=frac, random_state=seed)
    return raw.loc[raw["game_pk"].isin(games)]


def to_numpy_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Turn pybaseball's nullable number columns (missing = <NA>) into plain floats (NaN).

    Real Statcast has gaps, e.g. a batted ball with no tracked exit velocity, and <NA>
    values break ordinary comparisons and integer conversions downstream.
    """
    out = df.copy()
    for col in out.columns:
        dtype = out[col].dtype
        if isinstance(dtype, pd.api.extensions.ExtensionDtype) and (
            pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)
        ):
            out[col] = out[col].astype("float64")
    return out


def prepare(raw: pd.DataFrame, day_night: pd.Series | None = None) -> SeasonData:
    """Build the pitch table and plate-appearance tables from raw Statcast rows.

    ``day_night`` maps game_pk -> 1.0 (night) / 0.0 (day); unknown games get 0.5.
    """
    keep = [c for c in (*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS) if c in raw.columns]
    p = to_numpy_dtypes(raw.loc[raw["game_type"].eq("R"), keep])
    p["game_date"] = pd.to_datetime(p["game_date"]).dt.normalize()
    for col in ("game_pk", "batter", "pitcher", "at_bat_number", "pitch_number", "inning"):
        p[col] = p[col].astype("int64")
    # Statcast occasionally repeats a pitch row; keep one copy of each pitch.
    p = p.drop_duplicates(["game_pk", "at_bat_number", "pitch_number"], keep="last")
    p = p.sort_values(["game_pk", "at_bat_number", "pitch_number"], kind="stable")

    bottom = p["inning_topbot"].eq("Bot")
    p["bat_team"] = np.where(bottom, p["home_team"], p["away_team"])
    p["fld_team"] = np.where(bottom, p["away_team"], p["home_team"])
    p["family"] = p["pitch_type"].map(PITCH_FAMILIES)
    p["is_swing"] = p["description"].isin(SWING_DESCRIPTIONS)
    p["is_whiff"] = p["description"].isin(WHIFF_DESCRIPTIONS)

    # Pitches thrown in each plate appearance (for the pitcher's running pitch count).
    n_pitches = p.groupby(["game_pk", "at_bat_number"]).size().rename("n_pitches")

    pa_events = set(EVENT_TO_OUTCOME) | set(UNMODELED_PA_EVENTS)
    pa = p.loc[p["events"].isin(pa_events)].drop_duplicates(["game_pk", "at_bat_number"],
                                                            keep="last")  # fmt: skip
    pa = pa.join(n_pitches, on=["game_pk", "at_bat_number"]).reset_index(drop=True)
    pa = pa.sort_values(["game_pk", "at_bat_number"], kind="stable").reset_index(drop=True)

    # In-game context, computed over every plate appearance (including unmodeled ones).
    faced = pa.groupby(["game_pk", "pitcher"]).cumcount()
    pa["tto"] = np.minimum(faced // 9 + 1, 4)
    pa["batter_pa_num"] = np.minimum(pa.groupby(["game_pk", "batter"]).cumcount() + 1, 5)
    pa["pitcher_pitch_count"] = (
        pa.groupby(["game_pk", "pitcher"])["n_pitches"].cumsum() - pa["n_pitches"]
    )
    pa["is_home"] = pa["inning_topbot"].eq("Bot").astype(int)
    pa["park"] = pa["home_team"]
    pa["outs_made"] = pa["events"].map(OUTS_ON_EVENT).fillna(0).astype(int)
    if day_night is not None:
        pa["is_night"] = pa["game_pk"].map(day_night).fillna(0.5)
    else:
        pa["is_night"] = 0.5

    # Outcome columns for modeled plate appearances.
    pa["outcome"] = pa["events"].map(EVENT_TO_OUTCOME).fillna(-1).astype(int)
    pa_all = pa
    m = pa.loc[pa["outcome"] >= 0].copy()
    outcome = m["outcome"].to_numpy()
    m["is_hit"] = np.isin(outcome, HIT_CODES).astype(int)
    m["on_base"] = (np.isin(outcome, HIT_CODES) | (outcome == 1)).astype(int)
    m["tb"] = np.take(TOTAL_BASES, outcome)
    m["woba"] = np.take(WOBA_WEIGHTS, outcome)
    m["bbe"] = ((outcome >= 2) & (outcome <= OUT)).astype(int)
    m["hard"] = (m["bbe"].eq(1) & m["launch_speed"].ge(HARD_HIT_MPH)).astype(int)

    logger.info("Plate appearances: %s modeled of %s total, %s games",
                f"{len(m):,}", f"{len(pa_all):,}", f"{pa_all['game_pk'].nunique():,}")  # fmt: skip
    return SeasonData(pitches=p, pa_all=pa_all, pa=m.reset_index(drop=True))


def actual_team_runs(raw: pd.DataFrame) -> pd.Series | None:
    """Runs scored per team-game in the regular season, if score columns are available."""
    if "post_bat_score" not in raw.columns:
        return None
    r = raw.loc[raw["game_type"].eq("R")]
    bat_team = np.where(r["inning_topbot"].eq("Bot"), r["home_team"], r["away_team"])
    return r.assign(bat_team=bat_team).groupby(["game_pk", "bat_team"])["post_bat_score"].max()
