"""Point-in-time matchup features.

Every feature for a plate appearance on date D uses only games played *before* D, so the
backtest can't peek at the future. Training rows and live projections go through the same
``FeatureBuilder.transform`` call, so the model sees identical inputs in both cases.

Small samples are handled with shrinkage: each rate is a blend of the player's own record
and a sensible default, weighted by how much evidence the player has.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from matchup.config import (
    BATTER_SHRINK,
    BVP_SHRINK,
    FAMILIES,
    FAMILY_PA_SHRINK,
    FAMILY_SWING_SHRINK,
    HARD_HIT_SHRINK,
    HOT_COLD_DAYS,
    HR,
    OUTCOMES,
    PARK_SHRINK,
    PITCHER_SHRINK,
    RECENT_SHRINK,
    SPLIT_SHRINK,
    USAGE_SHRINK,
)
from matchup.data import SeasonData

logger = logging.getLogger(__name__)

RATE = [o for o in OUTCOMES if o != "OUT"]  # OUT is implied (rates sum to 1)
QUERY_COLUMNS = ("game_date", "batter", "pitcher", "stand", "p_throws", "tto", "batter_pa_num",
                 "pitcher_pitch_count", "is_home", "is_night", "park")  # fmt: skip

FEATURE_GROUPS: dict[str, list[str]] = {
    "Player skill & platoon": [
        *[f"bat_{o}" for o in RATE], *[f"pit_{o}" for o in RATE],
        *[f"log5_{o}" for o in OUTCOMES], "bat_pa_log", "pit_bf_log", "same_hand",
    ],
    "Pitch mix matchup": [
        *[f"usage_{f}" for f in FAMILIES], *[f"bat_whiff_{f}" for f in FAMILIES],
        *[f"bat_hit_{f}" for f in FAMILIES], *[f"pit_whiff_{f}" for f in FAMILIES],
        "mix_whiff", "mix_hit",
    ],
    "Contact quality": ["bat_hardhit", "pit_hardhit"],
    "Batter vs pitcher history": ["bvp_pa_log", "bvp_onbase_delta", "bvp_k_delta"],
    "Times through order": ["tto", "batter_pa_num", "pitcher_pitch_count"],
    "Home / away": ["is_home"],
    "Day / night": ["is_night"],
    "Park": ["park_hr_factor", "park_hit_factor"],
    "Hot / cold": ["bat_recent_ob_delta", "bat_recent_hard_delta", "pit_recent_k_delta",
                   "bat_recent_pa_log"],
}  # fmt: skip
FEATURES: list[str] = [f for cols in FEATURE_GROUPS.values() for f in cols]


def _cumulative(df: pd.DataFrame, keys: list[str], cols: list[str]) -> pd.DataFrame:
    """Per-key running totals by date (inclusive), ready for as-of lookups."""
    daily = df.groupby([*keys, "game_date"], sort=False)[cols].sum().reset_index()
    daily = daily.sort_values([*keys, "game_date"], kind="stable")
    daily[cols] = daily.groupby(keys, sort=False)[cols].cumsum()
    return daily.sort_values("game_date", kind="stable").reset_index(drop=True)


def _asof(queries: pd.DataFrame, table: pd.DataFrame, keys: list[str], cols: list[str],
          lag_days: int = 0) -> np.ndarray:  # fmt: skip
    """Totals for each query row from games strictly before (query date - lag_days)."""
    q = queries[keys].copy()
    q["game_date"] = queries["game_date"] - pd.Timedelta(days=lag_days)
    q["_row"] = np.arange(len(q))
    q = q.sort_values("game_date", kind="stable")
    merged = pd.merge_asof(q, table[[*keys, "game_date", *cols]], on="game_date", by=keys,
                           allow_exact_matches=False, direction="backward")  # fmt: skip
    merged = merged.sort_values("_row", kind="stable")
    return merged[cols].fillna(0.0).to_numpy(dtype=float)


def _shrink(count, n, prior, k):
    return (count + k * prior) / (n + k)


class FeatureBuilder:
    """Builds matchup features from a season of plate appearances and pitches.

    ``reference_end`` bounds the data used for league-wide baselines (league rates, park
    factors), so a backtest fold only uses league context from before its cutoff.
    """

    def __init__(self, season: SeasonData, reference_end: pd.Timestamp | str):
        self.reference_end = pd.Timestamp(reference_end)
        pa, pitches = season.pa, season.pitches
        ref = pa.loc[pa["game_date"] < self.reference_end]
        if ref.empty:
            ref = pa
        ref_pitches = pitches.loc[pitches["game_date"] < self.reference_end]
        if ref_pitches.empty:
            ref_pitches = pitches

        onehot = pd.get_dummies(pa["outcome"]).reindex(columns=range(7), fill_value=0)
        counts = [f"n_{o}" for o in OUTCOMES]
        pa = pa.assign(**{c: onehot[i].to_numpy(dtype=float) for i, c in enumerate(counts)},
                       one=1.0)  # fmt: skip

        # League baselines (reference window only).
        self.league = np.bincount(ref["outcome"], minlength=7) / len(ref)
        self.league_ob = float(ref["on_base"].mean())
        self.league_hard = float(ref["hard"].sum() / max(ref["bbe"].sum(), 1))

        # Park factors (reference window only), shrunk toward neutral.
        park = ref.groupby("park").agg(n=("outcome", "size"),
                                       hr=("outcome", lambda s: (s == HR).sum()),
                                       hit=("is_hit", "sum"))  # fmt: skip
        lg_hr, lg_hit = self.league[HR], float(ref["is_hit"].mean())
        self.park_hr = (_shrink(park["hr"], park["n"], lg_hr, PARK_SHRINK) / lg_hr).to_dict()
        self.park_hit = (_shrink(park["hit"], park["n"], lg_hit, PARK_SHRINK) / lg_hit).to_dict()

        # Running totals for as-of lookups (all dates; lookups only see earlier games).
        base = [*counts, "one"]
        self.bat_all = _cumulative(pa, ["batter"], [*base, "on_base", "bbe", "hard"])
        self.bat_split = _cumulative(pa, ["batter", "p_throws"], base)
        self.pit_all = _cumulative(pa, ["pitcher"], [*base, "bbe", "hard"])
        self.pit_split = _cumulative(pa, ["pitcher", "stand"], base)
        self.bvp = _cumulative(pa, ["batter", "pitcher"], ["one", "on_base", "n_K"])

        # Pitch-family tables.
        fam = pitches.dropna(subset=["family"])
        wide = pd.DataFrame({"game_date": fam["game_date"], "batter": fam["batter"],
                             "pitcher": fam["pitcher"], "stand": fam["stand"]})  # fmt: skip
        for f in FAMILIES:
            is_f = fam["family"].eq(f)
            wide[f"n_{f}"] = is_f.astype(float)
            wide[f"sw_{f}"] = (is_f & fam["is_swing"]).astype(float)
            wide[f"wh_{f}"] = (is_f & fam["is_whiff"]).astype(float)
        wide["n_all"] = 1.0
        fam_cols = [f"{p}_{f}" for f in FAMILIES for p in ("n", "sw", "wh")]
        self.pit_fam = _cumulative(wide, ["pitcher", "stand"], [*fam_cols, "n_all"])
        self.bat_fam_sw = _cumulative(wide, ["batter"], fam_cols)

        end = pa.dropna(subset=["family"])
        pa_fam = pd.DataFrame({"game_date": end["game_date"], "batter": end["batter"]})
        for f in FAMILIES:
            is_f = end["family"].eq(f)
            pa_fam[f"pa_{f}"] = is_f.astype(float)
            pa_fam[f"hit_{f}"] = (is_f & end["is_hit"].eq(1)).astype(float)
        self.bat_fam_pa = _cumulative(pa_fam, ["batter"], [c for c in pa_fam if c[:3] in ("pa_", "hit")])

        # League family baselines (reference window).
        rf = ref_pitches.dropna(subset=["family"])
        self.lg_whiff = {f: _safe_div(rf.loc[rf["family"].eq(f), "is_whiff"].sum(),
                                      rf.loc[rf["family"].eq(f), "is_swing"].sum())
                         for f in FAMILIES}  # fmt: skip
        re_ = ref.dropna(subset=["family"])
        self.lg_hit_fam = {f: float(re_.loc[re_["family"].eq(f), "is_hit"].mean())
                           if re_["family"].eq(f).any() else lg_hit for f in FAMILIES}  # fmt: skip
        usage = rf.groupby("stand")["family"].value_counts(normalize=True)
        self.lg_usage = {s: {f: float(usage.get((s, f), 1 / 3)) for f in FAMILIES}
                         for s in ("L", "R")}  # fmt: skip

    # ------------------------------------------------------------------ transform

    def transform(self, queries: pd.DataFrame) -> pd.DataFrame:
        """Features for each query row (a batter, a pitcher, a date and game context)."""
        missing = set(QUERY_COLUMNS) - set(queries.columns)
        if missing:
            raise ValueError(f"queries missing columns: {sorted(missing)}")
        q = queries.reset_index(drop=True).copy()
        q["game_date"] = pd.to_datetime(q["game_date"]).dt.normalize()
        for col in ("batter", "pitcher"):
            q[col] = q[col].astype("int64")
        n = len(q)
        out: dict[str, np.ndarray] = {}
        counts = [f"n_{o}" for o in OUTCOMES]
        lg = self.league

        # Batter: overall rates shrink to league, platoon split shrinks to overall.
        b_all = _asof(q, self.bat_all, ["batter"], [*counts, "one", "on_base", "bbe", "hard"])
        b_split = _asof(q, self.bat_split, ["batter", "p_throws"], [*counts, "one"])
        bat = self._split_rates(b_all[:, :7], b_all[:, 7], b_split[:, :7], b_split[:, 7],
                                BATTER_SHRINK)  # fmt: skip

        p_all = _asof(q, self.pit_all, ["pitcher"], [*counts, "one", "bbe", "hard"])
        p_split = _asof(q, self.pit_split, ["pitcher", "stand"], [*counts, "one"])
        pit = self._split_rates(p_all[:, :7], p_all[:, 7], p_split[:, :7], p_split[:, 7],
                                PITCHER_SHRINK)  # fmt: skip

        log5 = bat * pit / lg
        log5 /= log5.sum(axis=1, keepdims=True)
        for i, o in enumerate(OUTCOMES):
            if o != "OUT":
                out[f"bat_{o}"], out[f"pit_{o}"] = bat[:, i], pit[:, i]
        for i, o in enumerate(OUTCOMES):
            out[f"log5_{o}"] = log5[:, i]
        out["bat_pa_log"] = np.log1p(b_all[:, 7])
        out["pit_bf_log"] = np.log1p(p_all[:, 7])
        out["same_hand"] = (q["stand"].to_numpy() == q["p_throws"].to_numpy()).astype(float)

        # Pitch mix: pitcher's usage vs this batter's side x batter's results by family.
        fam_cols = [f"{p}_{f}" for f in FAMILIES for p in ("n", "sw", "wh")]
        pf = _asof(q, self.pit_fam, ["pitcher", "stand"], [*fam_cols, "n_all"])
        bs = _asof(q, self.bat_fam_sw, ["batter"], fam_cols)
        bp = _asof(q, self.bat_fam_pa, ["batter"],
                   [c for f in FAMILIES for c in (f"pa_{f}", f"hit_{f}")])  # fmt: skip
        stand = q["stand"].to_numpy()
        mix_whiff, mix_hit = np.zeros(n), np.zeros(n)
        for j, f in enumerate(FAMILIES):
            lg_use = np.where(stand == "L", self.lg_usage["L"][f], self.lg_usage["R"][f])
            usage = _shrink(pf[:, 3 * j], pf[:, -1], lg_use, USAGE_SHRINK)
            bat_whiff = _shrink(bs[:, 3 * j + 2], bs[:, 3 * j + 1], self.lg_whiff[f],
                                FAMILY_SWING_SHRINK)  # fmt: skip
            pit_whiff = _shrink(pf[:, 3 * j + 2], pf[:, 3 * j + 1], self.lg_whiff[f],
                                FAMILY_SWING_SHRINK)  # fmt: skip
            bat_hit = _shrink(bp[:, 2 * j + 1], bp[:, 2 * j], self.lg_hit_fam[f], FAMILY_PA_SHRINK)
            out[f"usage_{f}"], out[f"bat_whiff_{f}"] = usage, bat_whiff
            out[f"bat_hit_{f}"], out[f"pit_whiff_{f}"] = bat_hit, pit_whiff
            mix_whiff += usage * bat_whiff
            mix_hit += usage * bat_hit
        out["mix_whiff"], out["mix_hit"] = mix_whiff, mix_hit

        # Contact quality.
        out["bat_hardhit"] = _shrink(b_all[:, 10], b_all[:, 9], self.league_hard, HARD_HIT_SHRINK)
        out["pit_hardhit"] = _shrink(p_all[:, 9], p_all[:, 8], self.league_hard, HARD_HIT_SHRINK)

        # Batter vs pitcher history, as a heavily shrunk difference from the batter's norm.
        bvp = _asof(q, self.bvp, ["batter", "pitcher"], ["one", "on_base", "n_K"])
        bat_ob = bat[:, list(range(1, 6))].sum(axis=1)  # BB + hits
        out["bvp_pa_log"] = np.log1p(bvp[:, 0])
        out["bvp_onbase_delta"] = _shrink(bvp[:, 1], bvp[:, 0], bat_ob, BVP_SHRINK) - bat_ob
        out["bvp_k_delta"] = _shrink(bvp[:, 2], bvp[:, 0], bat[:, 0], BVP_SHRINK) - bat[:, 0]

        # In-game context supplied by the query.
        for col in ("tto", "batter_pa_num", "pitcher_pitch_count", "is_home", "is_night"):
            out[col] = q[col].to_numpy(dtype=float)
        park = q["park"].to_numpy()
        out["park_hr_factor"] = np.array([self.park_hr.get(p, 1.0) for p in park])
        out["park_hit_factor"] = np.array([self.park_hit.get(p, 1.0) for p in park])

        # Hot / cold: last HOT_COLD_DAYS days vs. season to date, shrunk.
        b_old = _asof(q, self.bat_all, ["batter"], ["one", "on_base", "bbe", "hard"],
                      lag_days=HOT_COLD_DAYS)  # fmt: skip
        r_pa, r_ob = b_all[:, 7] - b_old[:, 0], b_all[:, 8] - b_old[:, 1]
        r_bbe, r_hard = b_all[:, 9] - b_old[:, 2], b_all[:, 10] - b_old[:, 3]
        season_ob = _shrink(b_all[:, 8], b_all[:, 7], self.league_ob, 100)
        season_hard = out["bat_hardhit"]
        out["bat_recent_ob_delta"] = _shrink(r_ob, r_pa, season_ob, RECENT_SHRINK) - season_ob
        out["bat_recent_hard_delta"] = _shrink(r_hard, r_bbe, season_hard, RECENT_SHRINK) - season_hard
        out["bat_recent_pa_log"] = np.log1p(r_pa)
        p_old = _asof(q, self.pit_all, ["pitcher"], ["one", "n_K"], lag_days=HOT_COLD_DAYS)
        r_bf, r_k = p_all[:, 7] - p_old[:, 0], p_all[:, 0] - p_old[:, 1]
        out["pit_recent_k_delta"] = _shrink(r_k, r_bf, pit[:, 0], RECENT_SHRINK) - pit[:, 0]

        return pd.DataFrame(out, index=queries.index)[FEATURES]

    def _split_rates(self, all_counts, all_n, split_counts, split_n, shrink) -> np.ndarray:
        k = np.array([shrink[o] for o in OUTCOMES], dtype=float)
        overall = _shrink(all_counts, all_n[:, None], self.league, k)
        overall /= overall.sum(axis=1, keepdims=True)
        split = _shrink(split_counts, split_n[:, None], overall, SPLIT_SHRINK)
        return split / split.sum(axis=1, keepdims=True)


def _safe_div(a, b) -> float:
    return float(a) / float(b) if b else 0.0
