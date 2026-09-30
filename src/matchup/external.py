"""Day/night schedule and player names from MLB's public Stats API.

Statcast pitch data has neither game start times nor batter names. Both come from the
Stats API that powers Baseball Savant. Results are cached to ``data/``, and every call
fails soft: the pipeline keeps running with "unknown" values if the API is unreachable.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

STATS_API = "https://statsapi.mlb.com/api/v1"
TIMEOUT_S = 30


SCHEDULE_COLUMNS = ["game_pk", "date", "status", "day_night", "venue"]


def parse_schedule(payload: dict) -> pd.DataFrame:
    """Flatten a Stats API /schedule response into one row per listed game.

    Postponed and suspended games are listed more than once (on the original date and on
    the make-up date), so a game_pk can appear on several rows; ``day_night_map`` resolves it.
    """
    rows = []
    for day in payload.get("dates", []):
        for game in day.get("games", []):
            rows.append({
                "game_pk": int(game["gamePk"]),
                "date": day.get("date"),
                "status": (game.get("status") or {}).get("detailedState"),
                "day_night": str(game.get("dayNight", "")).lower() or None,
                "venue": (game.get("venue") or {}).get("name"),
            })  # fmt: skip
    return pd.DataFrame(rows, columns=SCHEDULE_COLUMNS)


def load_schedule(start: str, end: str, cache_path: Path, fetch: bool = True) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_csv(cache_path)
    if not fetch:
        return pd.DataFrame(columns=SCHEDULE_COLUMNS)
    try:
        import requests

        resp = requests.get(
            f"{STATS_API}/schedule",
            params={"sportId": 1, "startDate": start, "endDate": end, "gameType": "R"},
            timeout=TIMEOUT_S,
        )
        resp.raise_for_status()
        schedule = parse_schedule(resp.json())
    except Exception as exc:
        logger.warning("Could not fetch the schedule (%s); day/night will be 'unknown'.", exc)
        return pd.DataFrame(columns=SCHEDULE_COLUMNS)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(cache_path, index=False)
    logger.info("Schedule: %s games (%s night)", len(schedule),
                int((schedule["day_night"] == "night").sum()))  # fmt: skip
    return schedule


def day_night_map(schedule: pd.DataFrame) -> pd.Series:
    """game_pk -> 1.0 night / 0.0 day (other values are left out, i.e. unknown).

    A game listed more than once (postponed, suspended) keeps the listing where it was
    actually played: postponed/cancelled listings are dropped, then the latest one wins.
    """
    s = schedule.dropna(subset=["day_night"])
    s = s.loc[s["day_night"].isin(["day", "night"])]
    if "status" in s.columns:
        s = s.loc[~s["status"].fillna("").str.contains("Postponed|Cancelled", case=False)]
    s = s.drop_duplicates("game_pk", keep="last")
    return pd.Series((s["day_night"] == "night").astype(float).to_numpy(),
                     index=s["game_pk"].astype("int64"))  # fmt: skip


def parse_people(payload: dict) -> dict[int, str]:
    return {int(p["id"]): p["fullName"] for p in payload.get("people", []) if "fullName" in p}


def _stats_api_names(ids: list[int]) -> dict[int, str]:
    import requests

    names: dict[int, str] = {}
    for i in range(0, len(ids), 100):
        chunk = ids[i : i + 100]
        resp = requests.get(f"{STATS_API}/people",
                            params={"personIds": ",".join(map(str, chunk))}, timeout=TIMEOUT_S)
        resp.raise_for_status()
        names.update(parse_people(resp.json()))
    return names


def _chadwick_names(ids: list[int]) -> dict[int, str]:
    from pybaseball import playerid_reverse_lookup

    table = playerid_reverse_lookup(ids, key_type="mlbam")
    return {int(r.key_mlbam): f"{str(r.name_first).title()} {str(r.name_last).title()}"
            for r in table.itertuples()}  # fmt: skip


def pitcher_names_from_statcast(pitches: pd.DataFrame) -> dict[int, str]:
    """Statcast's ``player_name`` is the pitcher's name ("Last, First") in league-wide pulls.

    Used only as a last-resort fallback, and only if it really is constant per pitcher.
    """
    if "player_name" not in pitches.columns:
        return {}
    by_pitcher = pitches.dropna(subset=["player_name"]).groupby("pitcher")["player_name"]
    if by_pitcher.nunique().gt(1).mean() > 0.05:
        return {}  # not a per-pitcher name column; don't trust it
    out = {}
    for pid, raw in by_pitcher.first().items():
        last, _, first = str(raw).partition(", ")
        out[int(pid)] = f"{first} {last}".strip() if first else str(raw)
    return out


def load_player_names(
    ids: Iterable[int], cache_path: Path, pitches: pd.DataFrame, fetch: bool = True
) -> pd.DataFrame:
    """Names for every player ID, from cache, Stats API, Chadwick register, then Statcast."""
    ids = sorted({int(i) for i in ids})
    cached = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame(columns=["id", "name"])
    names = dict(zip(cached["id"].astype(int), cached["name"], strict=True))
    missing = [i for i in ids if i not in names]

    if missing and fetch:
        for source in (_stats_api_names, _chadwick_names):
            try:
                found = source(missing)
            except Exception as exc:
                logger.warning("Name lookup via %s failed (%s)", source.__name__, exc)
                continue
            names.update(found)
            missing = [i for i in missing if i not in names]
            if not missing:
                break
    if missing:
        for pid, name in pitcher_names_from_statcast(pitches).items():
            if pid in missing:
                names[pid] = name
        missing = [i for i in missing if i not in names]
    if missing:
        logger.warning("%d players have no name; use 'ID <number>' for them in game cards.",
                       len(missing))  # fmt: skip

    table = pd.DataFrame({"id": list(names), "name": list(names.values())})
    if fetch or not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(cache_path, index=False)
    return table
