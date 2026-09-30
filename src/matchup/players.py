"""Player directory: names, teams, handedness, roles, and name -> ID resolution."""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd

SUFFIXES = re.compile(r"\b(jr|sr|ii|iii|iv)\b")
ID_PATTERN = re.compile(r"^\s*(?:id\s*)?(\d{5,7})\s*$", re.IGNORECASE)
TEAM_SUFFIX = re.compile(r"^(.*?)\s*\(([A-Za-z]{2,3})\)\s*$")


def normalize(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = re.sub(r"[.'`’-]", " ", text)
    text = SUFFIXES.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


class ResolutionError(ValueError):
    """A game-card name couldn't be matched to exactly one player."""


@dataclass
class PlayerDirectory:
    players: pd.DataFrame  # id, name, norm, team, is_batter, is_pitcher, pa, bf, ...
    teams: frozenset[str]

    # ------------------------------------------------------------------ construction

    @classmethod
    def build(cls, pa_all: pd.DataFrame, names: pd.DataFrame,
              as_of: pd.Timestamp | None = None) -> PlayerDirectory:  # fmt: skip
        hist = pa_all if as_of is None else pa_all.loc[pa_all["game_date"] < as_of]
        if hist.empty:
            hist = pa_all

        last = hist.sort_values("game_date")
        bat = last.groupby("batter").agg(
            bat_team=("bat_team", "last"), pa=("batter", "size"),
            stands=("stand", lambda s: "".join(sorted(set(s)))),
            bat_last=("game_date", "max"),
        )  # fmt: skip
        pit = last.groupby("pitcher").agg(
            fld_team=("fld_team", "last"), bf=("pitcher", "size"),
            throws=("p_throws", lambda s: s.mode().iat[0]), pit_last=("game_date", "max"),
        )  # fmt: skip
        starts = starter_table(hist)
        pit = pit.join(
            starts.groupby("starter").agg(starts=("game_pk", "size"),
                                          avg_start_outs=("outs", "mean"))  # fmt: skip
        )
        df = bat.join(pit, how="outer")
        df.index.name = "id"
        df = df.reset_index()
        df["team"] = np.where(
            df["pit_last"].fillna(pd.Timestamp.min) > df["bat_last"].fillna(pd.Timestamp.min),
            df["fld_team"], df["bat_team"],
        )  # fmt: skip
        df["is_batter"] = df["pa"].fillna(0) > 0
        df["is_pitcher"] = df["bf"].fillna(0) > 0
        df[["pa", "bf", "starts"]] = df[["pa", "bf", "starts"]].fillna(0).astype(int)
        name_map = dict(zip(names["id"].astype(int), names["name"], strict=True))
        df["name"] = [name_map.get(int(i), f"ID {int(i)}") for i in df["id"]]
        df["norm"] = df["name"].map(normalize)
        teams = frozenset(pa_all["home_team"].unique()) | frozenset(pa_all["away_team"].unique())
        return cls(df.set_index("id", drop=False), teams)

    # ------------------------------------------------------------------ lookups

    def name(self, pid: int) -> str:
        return str(self.players.at[pid, "name"]) if pid in self.players.index else f"ID {pid}"

    def team(self, pid: int) -> str | None:
        return self.players.at[pid, "team"] if pid in self.players.index else None

    def throws(self, pid: int) -> str:
        if pid in self.players.index and isinstance(self.players.at[pid, "throws"], str):
            return self.players.at[pid, "throws"]
        return "R"

    def stand_vs(self, batter: int, p_throws: str) -> str:
        """Batting side against a pitcher; switch hitters take the platoon advantage."""
        stands = self.players.at[batter, "stands"] if batter in self.players.index else None
        if not isinstance(stands, str) or not stands:
            return "R"
        if len(stands) > 1:
            return "L" if p_throws == "R" else "R"
        return stands

    def avg_start_outs(self, pid: int, default: float = 15.0) -> float:
        if pid in self.players.index:
            value = self.players.at[pid, "avg_start_outs"]
            if pd.notna(value) and self.players.at[pid, "starts"] >= 2:
                return float(value)
        return default

    def search(self, query: str, limit: int = 10) -> pd.DataFrame:
        q = normalize(query)
        hits = self.players.loc[self.players["norm"].str.contains(q, regex=False)]
        return hits.sort_values(["pa", "bf"], ascending=False).head(limit)[
            ["id", "name", "team", "pa", "bf", "throws", "stands"]
        ]

    # ------------------------------------------------------------------ resolution

    def resolve(self, entry, team_hint: str | None = None, role: str | None = None) -> int:
        """Turn a game-card entry into an MLBAM ID.

        Accepts an integer ID, "ID 123456", "Name", "Name (TEAM)", or {name:, id:}.
        ``role`` ("batter"/"pitcher") breaks ties between same-named players.
        """
        if isinstance(entry, dict):
            if entry.get("id") is not None:
                return int(entry["id"])
            entry = entry.get("name", "")
        if isinstance(entry, (int, np.integer)):
            return int(entry)
        text = str(entry).strip()
        if m := ID_PATTERN.match(text):
            return int(m.group(1))
        if m := TEAM_SUFFIX.match(text):
            text, team_hint = m.group(1), m.group(2).upper()

        candidates = self.players.loc[self.players["norm"] == normalize(text)]
        if len(candidates) > 1 and role:
            by_role = candidates.loc[candidates[f"is_{role}"]]
            candidates = by_role if len(by_role) else candidates
        if len(candidates) > 1 and team_hint:
            on_team = candidates.loc[candidates["team"] == team_hint]
            candidates = on_team if len(on_team) else candidates
        if len(candidates) == 1:
            return int(candidates.index[0])
        if len(candidates) > 1:
            options = ", ".join(f"{r.name} ({r.team}, ID {r.id})" for r in candidates.itertuples())
            raise ResolutionError(
                f"'{entry}' matches several players: {options}. Write it as "
                f"'Name (TEAM)' or 'ID <number>'."
            )
        close = difflib.get_close_matches(normalize(text), self.players["norm"].tolist(), n=3,
                                          cutoff=0.75)  # fmt: skip
        hint = ""
        if close:
            names = self.players.loc[self.players["norm"].isin(close), "name"].unique()
            hint = f" Did you mean: {', '.join(names)}?"
        raise ResolutionError(f"No 2026 player named '{entry}'.{hint}")


def starter_table(pa_all: pd.DataFrame) -> pd.DataFrame:
    """One row per team-game: the starting pitcher and the outs he recorded."""
    first = pa_all.sort_values("at_bat_number").groupby(["game_pk", "fld_team"]).first()
    starters = first[["pitcher", "game_date"]].rename(columns={"pitcher": "starter"}).reset_index()
    outs = pa_all.groupby(["game_pk", "fld_team", "pitcher"])["outs_made"].sum()
    starters["outs"] = [
        outs.get((g, t, s), 0) for g, t, s in starters[["game_pk", "fld_team", "starter"]].itertuples(index=False)
    ]
    return starters
