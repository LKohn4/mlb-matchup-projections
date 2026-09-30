"""Turn a game card into matchup probabilities and a simulated projection."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from matchup.config import (
    HIT_CODES,
    OPENER_STAGE_CONTEXT,
    OUTCOMES,
    RELIEVER_STAGE_CONTEXT,
    STARTER_STAGE_CONTEXT,
    WOBA_WEIGHTS,
)
from matchup.gamecard import CardError, GameCard
from matchup.players import PlayerDirectory, ResolutionError
from matchup.simulate import N_SLOTS, N_STAGES, SLOT_NAMES, SimResult, StaffPlan, simulate_games

logger = logging.getLogger(__name__)
TIME_TO_NIGHT = {"night": 1.0, "day": 0.0, "unknown": 0.5}


@dataclass
class ResolvedTeam:
    team: str
    lineup: list[int]
    bench: list[int]
    staff: list[int | None]  # opener, starter, reliever 1-3, closer
    plan: StaffPlan


@dataclass
class GameProjection:
    card: GameCard
    teams: tuple[ResolvedTeam, ResolvedTeam]
    sim: SimResult
    per_pa: np.ndarray  # (2, 9, 6, 4, 7)
    bench_pa: dict[int, np.ndarray]  # team -> (n_bench, 6, 7)
    directory: PlayerDirectory
    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ tables

    def batter_lines(self) -> pd.DataFrame:
        table = self.sim.batter_table()
        woba = (self.sim.batting * np.asarray(WOBA_WEIGHTS)).sum(axis=-1).sum(axis=0)
        pa = self.sim.batting.sum(axis=-1).sum(axis=0)
        table["xwoba"] = [woba[t, s] / max(pa[t, s], 1) for t, s in
                          zip(table["team"], table["slot"] - 1, strict=True)]  # fmt: skip
        table["id"] = [self.teams[t].lineup[s - 1] for t, s in
                       zip(table["team"], table["slot"], strict=True)]  # fmt: skip
        table["name"] = table["id"].map(self.directory.name)
        table["team_code"] = table["team"].map(lambda t: self.teams[t].team)
        return table

    def pitcher_lines(self) -> pd.DataFrame:
        table = self.sim.pitcher_table()
        table["id"] = [self.teams[s].staff[k] for s, k in zip(table["staff"], table["slot"], strict=True)]
        table = table.dropna(subset=["id"]).copy()
        table["id"] = table["id"].astype(int)
        table["name"] = table["id"].map(self.directory.name)
        table["team_code"] = table["staff"].map(lambda t: self.teams[t].team)
        return table

    def pa_vs_pitchers(self) -> pd.DataFrame:
        """Expected plate appearances for each batter against each opposing pitcher."""
        rows = []
        for t in (0, 1):
            staff = self.teams[1 - t].staff
            for s in range(9):
                for k in range(N_SLOTS):
                    if staff[k] is None:
                        continue
                    rows.append({"team": self.teams[t].team, "slot": s + 1,
                                 "batter": self.directory.name(self.teams[t].lineup[s]),
                                 "pitcher": self.directory.name(staff[k]), "role": SLOT_NAMES[k],
                                 "expected_pa": self.sim.pa_vs[t, s, k] / self.sim.n_sims})  # fmt: skip
        return pd.DataFrame(rows)

    def matchup_matrix(self, team: int, metric: str = "xwoba", stage: int = 0) -> pd.DataFrame:
        """Per-PA metric for each batter (rows) vs each opposing pitcher (columns)."""
        staff = self.teams[1 - team].staff
        cols = [k for k in range(N_SLOTS) if staff[k] is not None]
        data = {}
        for k in cols:
            P = self.per_pa[team, :, k, stage]
            data[f"{self.directory.name(staff[k])}\n({SLOT_NAMES[k]})"] = _metric(P, metric)
        index = [f"{s + 1}. {self.directory.name(b)}" for s, b in enumerate(self.teams[team].lineup)]
        return pd.DataFrame(data, index=index)

    def reliever_scenarios(self, team: int) -> pd.DataFrame:
        """Each batter vs each available reliever option: per-PA hit, K and HR chances."""
        staff = self.teams[1 - team].staff
        rows = []
        for k in (2, 3, 4, 5):
            if staff[k] is None:
                continue
            for s, b in enumerate(self.teams[team].lineup):
                P = self.per_pa[team, s, k, 0]
                rows.append({"batter": f"{s + 1}. {self.directory.name(b)}",
                             "reliever": self.directory.name(staff[k]), "role": SLOT_NAMES[k],
                             "p_hit": float(P[list(HIT_CODES)].sum()), "p_k": float(P[0]),
                             "p_hr": float(P[5]), "xwoba": float(P @ np.asarray(WOBA_WEIGHTS))})  # fmt: skip
        return pd.DataFrame(rows)

    def bench_table(self) -> pd.DataFrame:
        rows = []
        for t, P in self.bench_pa.items():
            staff = self.teams[1 - t].staff
            for i, b in enumerate(self.teams[t].bench):
                for k in range(N_SLOTS):
                    if staff[k] is None:
                        continue
                    p = P[i, k]
                    rows.append({"team": self.teams[t].team, "batter": self.directory.name(b),
                                 "pitcher": self.directory.name(staff[k]), "role": SLOT_NAMES[k],
                                 "p_hit": float(p[list(HIT_CODES)].sum()), "p_k": float(p[0]),
                                 "xwoba": float(p @ np.asarray(WOBA_WEIGHTS))})  # fmt: skip
        return pd.DataFrame(rows)


def _metric(P: np.ndarray, metric: str) -> np.ndarray:
    if metric == "xwoba":
        return P @ np.asarray(WOBA_WEIGHTS)
    if metric == "p_hit":
        return P[:, list(HIT_CODES)].sum(axis=1)
    return P[:, OUTCOMES.index(metric)]


# ---------------------------------------------------------------------- resolution


def resolve_card(card: GameCard, directory: PlayerDirectory) -> tuple[tuple[ResolvedTeam, ResolvedTeam], list[str]]:
    errors, warnings, resolved = [], [], []
    for side, tc in (("away", card.away), ("home", card.home)):
        if tc.team not in directory.teams:
            errors.append(f"{side}: unknown team '{tc.team}'. Valid: {', '.join(sorted(directory.teams))}")

        def get(entry, role, where, _tc=tc, _side=side):
            try:
                pid = directory.resolve(entry, team_hint=_tc.team, role=role)
            except ResolutionError as exc:
                errors.append(f"{_side} {where}: {exc}")
                return None
            info = directory.players.loc[pid] if pid in directory.players.index else None
            if info is None or (role == "pitcher" and info["bf"] == 0) or (role == "batter" and info["pa"] == 0):
                warnings.append(f"{directory.name(pid)} has no 2026 {role} data; using league-average "
                                "rates.")  # fmt: skip
            return pid

        lineup = [get(p, "batter", f"lineup #{i + 1}") for i, p in enumerate(tc.lineup)]
        bench = [get(p, "batter", f"bench #{i + 1}") for i, p in enumerate(tc.bench)]
        starter = get(tc.starter, "pitcher", "starter")
        opener = get(tc.opener, "pitcher", "opener") if tc.opener else None
        relievers = [get(p, "pitcher", f"reliever #{i + 1}") for i, p in enumerate(tc.relievers)]
        closer = get(tc.closer, "pitcher", "closer") if tc.closer else None
        if len(set(lineup)) != len(lineup) and None not in lineup:
            errors.append(f"{side}: a batter appears twice in the lineup.")

        weights = np.zeros(3)
        if relievers:
            w = tc.reliever_weights or [1.0] * len(relievers)
            weights[: len(relievers)] = w
        starter_outs = (tc.starter_innings * 3 if tc.starter_innings
                        else directory.avg_start_outs(starter) if starter else 15.0)  # fmt: skip
        plan = StaffPlan(has_opener=opener is not None, opener_outs=max(1, round(tc.opener_innings * 3)),
                         starter_outs_mean=starter_outs, reliever_weights=tuple(weights),
                         has_closer=closer is not None)  # fmt: skip
        staff = [opener, starter, *(relievers + [None] * (3 - len(relievers))), closer]
        resolved.append(ResolvedTeam(tc.team, lineup, bench, staff, plan))
    if errors:
        raise CardError("Please fix the game card:\n  - " + "\n  - ".join(errors))
    return (resolved[0], resolved[1]), warnings


# ---------------------------------------------------------------------- queries


def _context(slot: int, stage: int) -> tuple[int, int, int]:
    table = (OPENER_STAGE_CONTEXT if slot == 0 else STARTER_STAGE_CONTEXT if slot == 1
             else RELIEVER_STAGE_CONTEXT)  # fmt: skip
    return table["tto"][stage], table["pa_num"][stage], table["pitches"][stage]


def build_queries(card: GameCard, teams: tuple[ResolvedTeam, ResolvedTeam],
                  directory: PlayerDirectory, key=None) -> pd.DataFrame:  # fmt: skip
    """One row per (batter, opposing pitcher slot, stage), plus bench rows at stage 0."""
    rows = []
    night = TIME_TO_NIGHT.get(card.time, 0.5)
    for t in (0, 1):
        staff = teams[1 - t].staff
        groups = [("lineup", teams[t].lineup, range(N_STAGES)), ("bench", teams[t].bench, [0])]
        for kind, batters, stages in groups:
            for i, b in enumerate(batters):
                for k, pitcher in enumerate(staff):
                    if pitcher is None:
                        continue
                    throws = directory.throws(pitcher)
                    for stage in stages:
                        tto, pa_num, pitches = _context(k, stage)
                        rows.append({
                            "key": key, "bat_side": t, "kind": kind, "idx": i, "slot": k,
                            "stage": stage, "game_date": pd.Timestamp(card.date), "batter": b,
                            "pitcher": pitcher, "stand": directory.stand_vs(b, throws),
                            "p_throws": throws, "tto": tto, "batter_pa_num": pa_num,
                            "pitcher_pitch_count": pitches, "is_home": t, "is_night": night,
                            "park": card.park,
                        })  # fmt: skip
    return pd.DataFrame(rows)


def assemble(queries: pd.DataFrame, P: np.ndarray, teams) -> tuple[np.ndarray, dict]:
    """Scatter predictions into the simulator's probability grid."""
    grid = np.zeros((2, 9, N_SLOTS, N_STAGES, len(OUTCOMES)))
    bench = {t: np.zeros((len(teams[t].bench), N_SLOTS, len(OUTCOMES))) for t in (0, 1)}
    q = queries.reset_index(drop=True)
    lu = (q["kind"] == "lineup").to_numpy()
    grid[q.loc[lu, "bat_side"], q.loc[lu, "idx"], q.loc[lu, "slot"], q.loc[lu, "stage"]] = P[lu]
    for t in (0, 1):
        m = ((q["kind"] == "bench") & (q["bat_side"] == t)).to_numpy()
        if m.any():
            bench[t][q.loc[m, "idx"], q.loc[m, "slot"]] = P[m]
    # Unused slots borrow the starter's row so the grid is always a valid distribution.
    for t in (0, 1):
        for k in range(N_SLOTS):
            if teams[1 - t].staff[k] is None:
                grid[t, :, k] = grid[t, :, 1]
    return grid, bench


def project_game(card: GameCard, directory: PlayerDirectory, builder, model,
                 sims: int | None = None, seed: int = 42) -> GameProjection:  # fmt: skip
    teams, warnings = resolve_card(card, directory)
    for w in warnings:
        logger.warning(w)
    queries = build_queries(card, teams, directory)
    P = model.predict_proba(builder.transform(queries))
    grid, bench = assemble(queries, P, teams)
    sim = simulate_games(grid, (teams[0].plan, teams[1].plan), n_sims=sims or card.sims or 10_000,
                         seed=seed)  # fmt: skip
    return GameProjection(card, teams, sim, grid, bench, directory, warnings)
