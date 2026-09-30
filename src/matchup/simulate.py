"""Monte Carlo game simulator.

Plays thousands of games at once (vectorized with NumPy). Each plate appearance is drawn
from the matchup model's probabilities for that batter, that pitcher and that point in the
game. The simulator supplies what a single-PA model can't: how often each batter comes up,
which pitcher he faces, and when the starter leaves.

Rules modeled: 9-inning games, home team skips the bottom of the 9th when ahead, walk-offs,
extra innings with a runner on second, openers, starters pulled at a sampled out count,
relievers chosen by the given weights each inning, and a closer in save situations.
Baserunning uses fixed MLB-typical advancement rates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from matchup.config import BB, DOUBLE, HR, OUT, SINGLE, TOTAL_BASES, TRIPLE, K

N_SLOTS, N_STAGES = 6, 4
OPENER, STARTER, CLOSER = 0, 1, 5
RELIEVERS = (2, 3, 4)
SLOT_NAMES = ("Opener", "Starter", "Reliever 1", "Reliever 2", "Reliever 3", "Closer")

# Typical MLB baserunning rates (approximations).
GIDP_RATE = 0.12  # in-play out with a runner on 1st and < 2 outs becomes a double play
SAC_SCORE_RATE = 0.35  # runner on 3rd scores on an in-play out with < 2 outs
OUT_ADVANCE_2B = 0.25  # runner on 2nd reaches 3rd on an in-play out
SINGLE_SCORES_FROM_2B = 0.60
SINGLE_1B_TO_3B = 0.28
DOUBLE_SCORES_FROM_1B = 0.42
REACH_ON_ERROR = 0.025  # share of in-play outs where the batter reaches on an error
STEAL_OR_WP_1B = 0.05  # runner on 1st takes 2nd (steal, wild pitch, passed ball) per PA
STEAL_OR_WP_2B = 0.02  # runner on 2nd takes 3rd


@dataclass
class StaffPlan:
    """How one team's pitchers are used."""

    has_opener: bool = False
    opener_outs: int = 3
    starter_outs_mean: float = 16.0
    starter_outs_sd: float = 3.0
    reliever_weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)  # 0 = unavailable
    has_closer: bool = True

    @property
    def weights(self) -> np.ndarray:
        w = np.asarray(self.reliever_weights, dtype=float)
        return w / w.sum() if w.sum() > 0 else w


@dataclass
class SimResult:
    n_sims: int
    batting: np.ndarray  # (N, 2, 9, 7) outcome counts per sim, team, lineup slot
    pa_vs: np.ndarray  # (2, 9, 6) total PAs per team, lineup slot, opposing pitcher slot
    pitching: np.ndarray  # (N, 2, 6, 7) outcomes allowed per sim, staff, pitcher slot
    pitch_outs: np.ndarray  # (N, 2, 6)
    pitch_runs: np.ndarray  # (N, 2, 6)
    score: np.ndarray  # (N, 2) final runs [away, home]
    innings: np.ndarray  # (N,)
    extras: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ summaries

    def batter_table(self) -> pd.DataFrame:
        """Expected line per lineup slot, plus chances of at least one hit/HR/K."""
        b = self.batting
        hits = b[..., SINGLE] + b[..., DOUBLE] + b[..., TRIPLE] + b[..., HR]
        tb = (b * np.asarray(TOTAL_BASES)).sum(axis=-1)
        rows = []
        for team in (0, 1):
            for slot in range(9):
                rows.append({
                    "team": team, "slot": slot + 1,
                    "pa": b[:, team, slot].sum(axis=-1).mean(),
                    "h": hits[:, team, slot].mean(), "1b": b[:, team, slot, SINGLE].mean(),
                    "2b": b[:, team, slot, DOUBLE].mean(), "3b": b[:, team, slot, TRIPLE].mean(),
                    "hr": b[:, team, slot, HR].mean(), "bb": b[:, team, slot, BB].mean(),
                    "k": b[:, team, slot, K].mean(), "tb": tb[:, team, slot].mean(),
                    "p_1plus_h": (hits[:, team, slot] >= 1).mean(),
                    "p_2plus_h": (hits[:, team, slot] >= 2).mean(),
                    "p_1plus_hr": (b[:, team, slot, HR] >= 1).mean(),
                    "p_1plus_k": (b[:, team, slot, K] >= 1).mean(),
                })  # fmt: skip
        return pd.DataFrame(rows)

    def pitcher_table(self) -> pd.DataFrame:
        p = self.pitching
        hits = p[..., SINGLE] + p[..., DOUBLE] + p[..., TRIPLE] + p[..., HR]
        bf = p.sum(axis=-1)
        rows = []
        for staff in (0, 1):
            for slot in range(N_SLOTS):
                rows.append({
                    "staff": staff, "slot": slot, "role": SLOT_NAMES[slot],
                    "p_appear": (bf[:, staff, slot] > 0).mean(),
                    "bf": bf[:, staff, slot].mean(), "ip": self.pitch_outs[:, staff, slot].mean() / 3,
                    "k": p[:, staff, slot, K].mean(), "bb": p[:, staff, slot, BB].mean(),
                    "h": hits[:, staff, slot].mean(), "hr": p[:, staff, slot, HR].mean(),
                    "r": self.pitch_runs[:, staff, slot].mean(),
                })  # fmt: skip
        return pd.DataFrame(rows)

    def game_summary(self) -> dict:
        away, home = self.score[:, 0], self.score[:, 1]
        decided = away != home
        return {
            "home_win_prob": float((home > away).sum() / max(decided.sum(), 1)),
            "away_runs": float(away.mean()), "home_runs": float(home.mean()),
            "total_runs": float((away + home).mean()),
            "p_extra_innings": float((self.innings > 9).mean()),
            "sims": self.n_sims,
        }  # fmt: skip


def simulate_games(probs: np.ndarray, plans: tuple[StaffPlan, StaffPlan], n_sims: int = 10_000,
                   seed: int = 42, max_innings: int = 15) -> SimResult:  # fmt: skip
    """Simulate ``n_sims`` games.

    probs: (2, 9, 6, 4, 7) outcome probabilities indexed by batting team (0 away, 1 home),
    lineup slot, opposing pitcher slot, stage (the pitcher's times through the order, 0-3)
    and outcome. ``plans[t]`` describes team t's pitching staff.
    """
    rng = np.random.default_rng(seed)
    cum = np.cumsum(probs, axis=-1)
    N = n_sims
    inning = np.ones(N, dtype=np.int64)
    half = np.zeros(N, dtype=np.int64)  # 0 = top (away bats), 1 = bottom (home bats)
    outs = np.zeros(N, dtype=np.int64)
    b1, b2, b3 = (np.zeros(N, dtype=bool) for _ in range(3))
    score = np.zeros((N, 2), dtype=np.int64)
    lineup = np.zeros((N, 2), dtype=np.int64)
    done = np.zeros(N, dtype=bool)
    new_half = np.ones(N, dtype=bool)

    cur = np.array([[OPENER if p.has_opener else STARTER for p in plans]] * N)
    p_outs = np.zeros((N, 2), dtype=np.int64)
    p_bf = np.zeros((N, 2), dtype=np.int64)
    target = np.stack([np.clip(np.round(rng.normal(p.starter_outs_mean, p.starter_outs_sd, N)),
                               1, 27) for p in plans], axis=1)  # fmt: skip
    opener_outs = np.array([p.opener_outs for p in plans])
    rel_w = [p.weights for p in plans]
    has_rel = [w.sum() > 0 for w in rel_w]
    has_closer = [p.has_closer for p in plans]

    batting = np.zeros((N, 2, 9, 7), dtype=np.int16)
    pitching = np.zeros((N, 2, N_SLOTS, 7), dtype=np.int16)
    pitch_outs = np.zeros((N, 2, N_SLOTS), dtype=np.int16)
    pitch_runs = np.zeros((N, 2, N_SLOTS), dtype=np.int16)
    pa_vs = np.zeros((2, 9, N_SLOTS))

    def bullpen(sel: np.ndarray, t: int) -> np.ndarray:
        lead = score[sel, t] - score[sel, 1 - t]
        slots = np.full(len(sel), STARTER)
        if has_rel[t]:
            slots = rng.choice(RELIEVERS, size=len(sel), p=rel_w[t])
        elif has_closer[t]:
            slots = np.full(len(sel), CLOSER)
        if has_closer[t]:
            save = (inning[sel] >= 9) & (lead >= 0) & (lead <= 3)
            slots = np.where(save, CLOSER, slots)
        return slots

    for _ in range(1000):  # safety bound; games finish long before this
        a = np.flatnonzero(~done)
        if a.size == 0:
            break
        bt = half[a]
        ft = 1 - bt

        # Pitching changes for the fielding team.
        for t in (0, 1):
            m = ft == t
            if not m.any():
                continue
            sel = a[m]
            slot = cur[sel, t]
            to_starter = sel[(slot == OPENER) & (p_outs[sel, t] >= opener_outs[t])]
            cur[to_starter, t] = STARTER
            p_outs[to_starter, t] = 0
            p_bf[to_starter, t] = 0
            slot = cur[sel, t]
            pull = (slot == STARTER) & (p_outs[sel, t] >= target[sel, t])
            new_inning_pen = new_half[sel] & (slot >= 2)
            need = sel[pull | new_inning_pen]
            if need.size:
                chosen = bullpen(need, t)
                changed = chosen != cur[need, t]
                cur[need, t] = chosen
                p_outs[need[changed], t] = 0
                p_bf[need[changed], t] = 0
        new_half[a] = False

        slot = cur[a, ft]
        stage = np.minimum(p_bf[a, ft] // 9, N_STAGES - 1)
        batter = lineup[a, bt]
        c = cum[bt, batter, slot, stage]
        outcome = np.minimum((rng.random(a.size)[:, None] > c).sum(axis=1), OUT)

        np.add.at(batting, (a, bt, batter, outcome), 1)
        np.add.at(pitching, (a, ft, slot, outcome), 1)
        np.add.at(pa_vs, (bt, batter, slot), 1)

        # Baserunning. First, advances during the plate appearance (steals, wild pitches).
        B1, B2, B3, outs_before = b1[a], b2[a], b3[a], outs[a]
        r = rng.random((a.size, 5))
        take3 = B2 & ~B3 & (r[:, 3] < STEAL_OR_WP_2B)
        B3, B2 = B3 | take3, B2 & ~take3
        take2 = B1 & ~B2 & (r[:, 4] < STEAL_OR_WP_1B)
        B2, B1 = B2 | take2, B1 & ~take2
        # An in-play out is sometimes an error: the batter reaches, runners move up like a walk.
        roe = (outcome == OUT) & (rng.random(a.size) < REACH_ON_ERROR)
        runs = np.zeros(a.size, dtype=np.int64)
        new_outs = outs_before + (outcome == K)
        nb1, nb2, nb3 = B1.copy(), B2.copy(), B3.copy()

        o = (outcome == OUT) & ~roe
        gidp = o & B1 & (outs_before < 2) & (r[:, 0] < GIDP_RATE)
        new_outs = new_outs + o + gidp
        sac = o & B3 & (outs_before < 2) & (r[:, 1] < SAC_SCORE_RATE) & (new_outs < 3)
        adv2 = o & B2 & ~B3 & (outs_before < 2) & (r[:, 2] < OUT_ADVANCE_2B) & (new_outs < 3)
        runs += sac
        nb3 = np.where(o, (B3 & ~sac) | adv2, nb3)
        nb2 = np.where(o, B2 & ~adv2, nb2)
        nb1 = np.where(o, B1 & ~gidp, nb1)

        w = (outcome == BB) | roe
        runs += w & B1 & B2 & B3
        nb3 = np.where(w, B3 | (B2 & B1), nb3)
        nb2 = np.where(w, B2 | B1, nb2)
        nb1 = np.where(w, True, nb1)

        s = outcome == SINGLE
        from2 = B2 & (r[:, 0] < SINGLE_SCORES_FROM_2B)
        runs += s & B3
        runs += s & from2
        hold3 = B2 & ~from2
        first_to_third = B1 & ~hold3 & (r[:, 1] < SINGLE_1B_TO_3B)
        nb3 = np.where(s, hold3 | first_to_third, nb3)
        nb2 = np.where(s, B1 & ~first_to_third, nb2)
        nb1 = np.where(s, True, nb1)

        d = outcome == DOUBLE
        from1 = B1 & (r[:, 2] < DOUBLE_SCORES_FROM_1B)
        runs += d & B3
        runs += d & B2
        runs += d & from1
        nb3 = np.where(d, B1 & ~from1, nb3)
        nb2 = np.where(d, True, nb2)
        nb1 = np.where(d, False, nb1)

        tr = outcome == TRIPLE
        runs += tr * (B1.astype(int) + B2 + B3)
        nb3 = np.where(tr, True, nb3)
        nb2 = np.where(tr, False, nb2)
        nb1 = np.where(tr, False, nb1)

        h = outcome == HR
        runs += h * (1 + B1.astype(int) + B2 + B3)
        nb1, nb2, nb3 = nb1 & ~h, nb2 & ~h, nb3 & ~h

        np.add.at(score, (a, bt), runs)
        np.add.at(pitch_runs, (a, ft, slot), runs)
        gained = new_outs - outs_before
        np.add.at(pitch_outs, (a, ft, slot), gained)
        p_outs[a, ft] += gained
        p_bf[a, ft] += 1
        lineup[a, bt] = (batter + 1) % 9
        outs[a], b1[a], b2[a], b3[a] = new_outs, nb1, nb2, nb3

        # Walk-off.
        walkoff = (bt == 1) & (inning[a] >= 9) & (score[a, 1] > score[a, 0])
        done[a[walkoff]] = True

        # End of half-inning.
        ended = a[(new_outs >= 3) & ~walkoff]
        if ended.size:
            top = half[ended] == 0
            home_ahead = score[ended, 1] > score[ended, 0]
            over_top = top & (inning[ended] >= 9) & home_ahead
            bottom = ~top
            over_bottom = bottom & (inning[ended] >= 9) & (score[ended, 0] != score[ended, 1])
            capped = bottom & (inning[ended] >= max_innings)
            finish = over_top | over_bottom | capped
            done[ended[finish]] = True
            cont = ended[~finish]
            was_top = half[cont] == 0
            half[cont] = np.where(was_top, 1, 0)
            inning[cont] += ~was_top
            outs[cont] = 0
            b1[cont] = False
            b3[cont] = False
            b2[cont] = inning[cont] >= 10  # extra-inning runner on second
            new_half[cont] = True

    return SimResult(n_sims=N, batting=batting, pa_vs=pa_vs, pitching=pitching,
                     pitch_outs=pitch_outs, pitch_runs=pitch_runs, score=score, innings=inning)
