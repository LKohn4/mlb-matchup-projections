"""Game cards: the scoresheet you fill in to project a game.

A card is a small YAML file with the date, park, day/night, and each team's lineup, bench
and pitching plan. ``generate_card`` fills one in automatically from 2026 usage (latest
lineup, next starter in the rotation, most-used relievers, closer), ready to edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from matchup.players import starter_table

TIMES = ("day", "night", "unknown")


class CardError(ValueError):
    """The game card has a problem the user needs to fix."""


@dataclass
class TeamCard:
    team: str
    lineup: list
    starter: object
    bench: list = field(default_factory=list)
    starter_innings: float | None = None
    opener: object | None = None
    opener_innings: float = 1.0
    relievers: list = field(default_factory=list)
    reliever_weights: list[float] | None = None
    closer: object | None = None


@dataclass
class GameCard:
    date: str
    park: str
    time: str
    away: TeamCard
    home: TeamCard
    sims: int | None = None
    title: str | None = None

    @property
    def slug(self) -> str:
        return f"{self.date}_{self.away.team}_at_{self.home.team}"


# ---------------------------------------------------------------------- reading


def _team(data: dict, side: str, errors: list[str]) -> TeamCard:
    if not isinstance(data, dict):
        errors.append(f"'{side}' must be a section with team, lineup and pitchers.")
        return TeamCard(team="?", lineup=[], starter=None)
    pitchers = data.get("pitchers") or {}
    team = str(data.get("team", "")).upper()
    lineup = list(data.get("lineup") or [])
    if not team:
        errors.append(f"{side}: 'team' is missing (e.g. NYY).")
    if len(lineup) != 9:
        errors.append(f"{side}: the lineup needs exactly 9 batters (found {len(lineup)}).")
    if not pitchers.get("starter"):
        errors.append(f"{side}: 'pitchers: starter' is required.")
    relievers = list(pitchers.get("relievers") or [])
    if len(relievers) > 3:
        errors.append(f"{side}: list at most 3 relievers (found {len(relievers)}).")
    weights = pitchers.get("reliever_weights")
    if weights is not None and len(weights) != len(relievers):
        errors.append(f"{side}: reliever_weights needs one number per reliever.")
    innings = pitchers.get("starter_innings")
    if innings is not None and not 0 < float(innings) <= 9:
        errors.append(f"{side}: starter_innings must be between 0 and 9.")
    return TeamCard(
        team=team, lineup=lineup, bench=list(data.get("bench") or []),
        starter=pitchers.get("starter"),
        starter_innings=float(innings) if innings is not None else None,
        opener=pitchers.get("opener") or None,
        opener_innings=float(pitchers.get("opener_innings", 1.0)),
        relievers=relievers,
        reliever_weights=[float(w) for w in weights] if weights is not None else None,
        closer=pitchers.get("closer") or None,
    )  # fmt: skip


def parse_card(data: dict) -> GameCard:
    errors: list[str] = []
    if not isinstance(data, dict):
        raise CardError("The game card is empty or isn't valid YAML.")
    game = data.get("game") or {}
    date = str(game.get("date", ""))
    try:
        pd.Timestamp(date)
    except (ValueError, TypeError):
        errors.append(f"game: date '{date}' isn't a date like 2026-10-02.")
    time = str(game.get("time", "unknown")).lower()
    if time not in TIMES:
        errors.append(f"game: time must be one of {TIMES} (found '{time}').")
    away = _team(data.get("away"), "away", errors)
    home = _team(data.get("home"), "home", errors)
    park = str(game.get("park") or home.team).upper()
    if errors:
        raise CardError("Please fix the game card:\n  - " + "\n  - ".join(errors))
    return GameCard(date=date, park=park, time=time, away=away, home=home,
                    sims=game.get("sims"), title=game.get("title"))  # fmt: skip


def load_card(path: Path) -> GameCard:
    path = Path(path)
    if not path.exists():
        raise CardError(f"Game card not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise CardError(f"{path.name} isn't valid YAML: {exc}") from exc
    return parse_card(data)


# ---------------------------------------------------------------------- writing


def _q(value) -> str:
    return '"' + str(value).replace('"', "'") + '"'


def _team_yaml(tc: TeamCard, label: str) -> str:
    lines = [f"{label}:", f"  team: {tc.team}", "  lineup:          # batting order, 1-9"]
    lines += [f"    - {_q(p)}" for p in tc.lineup]
    lines.append("  bench:           # optional; projected against every pitcher, not simulated")
    lines += [f"    - {_q(p)}" for p in tc.bench] or ["    []"]
    lines.append("  pitchers:")
    lines.append(f"    starter: {_q(tc.starter)}")
    if tc.starter_innings is not None:
        lines.append(f"    starter_innings: {tc.starter_innings:.1f}   # delete to use his 2026 average")
    lines.append(f"    opener: {_q(tc.opener) if tc.opener else 'null'}   # optional: pitches first, then the starter")
    lines.append(f"    opener_innings: {tc.opener_innings:.1f}")
    lines.append("    relievers:     # up to 3 available relievers")
    lines += [f"      - {_q(p)}" for p in tc.relievers] or ["      []"]
    if tc.reliever_weights:
        lines.append(f"    reliever_weights: [{', '.join(f'{w:.2f}' for w in tc.reliever_weights)}]"
                     "   # relative chance each is used")  # fmt: skip
    lines.append(f"    closer: {_q(tc.closer) if tc.closer else 'null'}")
    return "\n".join(lines)


def card_to_yaml(card: GameCard, note: str = "") -> str:
    header = [
        "# Game card for mlb-matchup-projections",
        "# Names must match 2026 players. If a name is shared, write it as \"Name (TEAM)\" or",
        "# use \"ID 123456\". Search with:  mlb-matchup players <name>",
    ]
    if note:
        header.append(f"# {note}")
    game = [
        "game:",
        f"  date: {card.date}      # projections use 2026 games played before this date",
        f"  park: {card.park}            # home team's park",
        f"  time: {card.time}          # day, night or unknown",
        f"  sims: {card.sims or 10000}         # simulated games",
    ]
    return "\n".join([*header, "", *game, "", _team_yaml(card.away, "away"), "",
                      _team_yaml(card.home, "home"), ""])  # fmt: skip


def save_card(card: GameCard, path: Path, note: str = "") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(card_to_yaml(card, note), encoding="utf-8")
    return path


# ---------------------------------------------------------------------- generation


def _latest_lineup(pa_all: pd.DataFrame, team: str, before: pd.Timestamp) -> tuple[int, list[int]]:
    games = pa_all.loc[(pa_all["bat_team"] == team) & (pa_all["game_date"] < before)]
    if games.empty:
        raise CardError(f"No 2026 games found for team '{team}' before {before.date()}.")
    last_game = games.loc[games["game_date"] == games["game_date"].max(), "game_pk"].max()
    return int(last_game), lineup_for_game(pa_all, int(last_game), team)


def lineup_for_game(pa_all: pd.DataFrame, game_pk: int, team: str) -> list[int]:
    rows = pa_all.loc[(pa_all["game_pk"] == game_pk) & (pa_all["bat_team"] == team)]
    order = rows.sort_values("at_bat_number")["batter"].drop_duplicates().tolist()
    return [int(b) for b in order[:9]]


def bullpen_usage(pa_all: pd.DataFrame, team: str, before: pd.Timestamp,
                  days: int = 30) -> tuple[list[int], int | None]:  # fmt: skip
    """Most-used relievers and the likely closer over the last ``days`` days."""
    window = pa_all.loc[(pa_all["fld_team"] == team) & (pa_all["game_date"] < before)
                        & (pa_all["game_date"] >= before - pd.Timedelta(days=days))]  # fmt: skip
    if window.empty:
        return [], None
    starters = set(starter_table(window)["starter"])
    pen = window.loc[~window["pitcher"].isin(starters)]
    if pen.empty:
        return [], None
    ninth = pen.loc[pen["inning"] >= 9].groupby("pitcher")["game_pk"].nunique()
    closer = int(ninth.idxmax()) if len(ninth) else None
    apps = pen.groupby("pitcher")["game_pk"].nunique().sort_values(ascending=False)
    relievers = [int(p) for p in apps.index if p != closer][:3]
    return relievers, closer


def next_starter(pa_all: pd.DataFrame, team: str, before: pd.Timestamp) -> int:
    """The pitcher who started five team-games ago: next up in a five-man rotation."""
    starts = starter_table(pa_all.loc[pa_all["game_date"] < before])
    starts = starts.loc[starts["fld_team"] == team].sort_values(["game_date", "game_pk"])
    if starts.empty:
        raise CardError(f"No 2026 starts found for team '{team}'.")
    return int(starts["starter"].iloc[-5] if len(starts) >= 5 else starts["starter"].iloc[-1])


def generate_card(pa_all: pd.DataFrame, directory, away: str, home: str, date: str,
                  time: str = "night", sims: int = 10_000) -> GameCard:  # fmt: skip
    """A ready-to-edit card for any two teams, built from 2026 usage before ``date``."""
    before = pd.Timestamp(date)
    teams = {}
    for team in (away.upper(), home.upper()):
        if team not in directory.teams:
            raise CardError(f"Unknown team '{team}'. Valid codes: {', '.join(sorted(directory.teams))}")
        _, lineup = _latest_lineup(pa_all, team, before)
        recent = pa_all.loc[(pa_all["bat_team"] == team) & (pa_all["game_date"] < before)
                            & (pa_all["game_date"] >= before - pd.Timedelta(days=30))]  # fmt: skip
        bench = [int(b) for b in recent["batter"].value_counts().index if b not in lineup][:4]
        starter = next_starter(pa_all, team, before)
        relievers, closer = bullpen_usage(pa_all, team, before)
        teams[team] = TeamCard(
            team=team, lineup=[_display(directory, p, team) for p in lineup],
            bench=[_display(directory, p, team) for p in bench],
            starter=_display(directory, starter, team),
            starter_innings=round(directory.avg_start_outs(starter) / 3, 1),
            relievers=[_display(directory, p, team) for p in relievers],
            closer=_display(directory, closer, team) if closer else None,
        )  # fmt: skip
    return GameCard(date=str(before.date()), park=home.upper(), time=time,
                    away=teams[away.upper()], home=teams[home.upper()], sims=sims)  # fmt: skip


def card_from_game(pa_all: pd.DataFrame, directory, game_pk: int, is_night: float) -> GameCard:
    """Rebuild a real game's card (actual lineups and starters, bullpen as known beforehand)."""
    rows = pa_all.loc[pa_all["game_pk"] == game_pk]
    date = rows["game_date"].iloc[0]
    home, away = rows["home_team"].iloc[0], rows["away_team"].iloc[0]
    teams = {}
    for team in (away, home):
        fielding = rows.loc[rows["fld_team"] == team].sort_values("at_bat_number")
        starter = int(fielding["pitcher"].iloc[0])
        relievers, closer = bullpen_usage(pa_all, team, date)
        relievers = [r for r in relievers if r != starter]
        teams[team] = TeamCard(
            team=team, lineup=[f"ID {b}" for b in lineup_for_game(pa_all, game_pk, team)],
            starter=f"ID {starter}", starter_innings=None,
            relievers=[f"ID {r}" for r in relievers],
            closer=f"ID {closer}" if closer and closer != starter else None,
        )  # fmt: skip
    time = "night" if is_night == 1 else ("day" if is_night == 0 else "unknown")
    return GameCard(date=str(date.date()), park=home, time=time, away=teams[away],
                    home=teams[home])  # fmt: skip


def _display(directory, pid: int, team: str) -> str:
    """Name as it should appear on a card: add the team if the name is shared."""
    name = directory.name(pid)
    if name.startswith("ID "):
        return name
    same = directory.players.loc[directory.players["norm"] == directory.players.at[pid, "norm"]]
    if len(same) > 1:
        on_team = same.loc[same["team"] == team]
        return f"{name} ({team})" if len(on_team) == 1 else f"ID {pid}"
    return name
