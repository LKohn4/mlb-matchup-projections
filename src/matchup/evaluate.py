"""Scoring for probability forecasts. Accuracy is deliberately not the headline metric:
most plate appearances end in an out, so a model can look "accurate" while being useless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from matchup.config import OUTCOMES, WOBA_WEIGHTS


def log_loss(y: np.ndarray, P: np.ndarray) -> float:
    """Average negative log-probability of what actually happened (lower is better)."""
    p = np.clip(P[np.arange(len(y)), y], 1e-12, 1)
    return float(-np.mean(np.log(p)))


def brier(y: np.ndarray, P: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error of the probability vector."""
    onehot = np.eye(P.shape[1])[y]
    return float(np.mean(np.sum((P - onehot) ** 2, axis=1)))


def expected_woba(P: np.ndarray) -> np.ndarray:
    return P @ np.asarray(WOBA_WEIGHTS)


def score_model(name: str, fold: str, y: np.ndarray, P: np.ndarray, train_time: float,
                baseline_ll: float | None = None) -> dict:  # fmt: skip
    ll = log_loss(y, P)
    row = {"model": name, "fold": fold, "n": len(y), "log_loss": ll, "brier": brier(y, P),
           "train_time_s": train_time}  # fmt: skip
    if baseline_ll is not None:
        row["skill_vs_league_pct"] = 100 * (baseline_ll - ll) / baseline_ll
    return row


def calibration_table(y: np.ndarray, P: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Predicted vs. actual frequency per outcome, in equal-count bins of predictions."""
    rows = []
    for k, outcome in enumerate(OUTCOMES):
        p = P[:, k]
        actual = (y == k).astype(float)
        edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            continue
        which = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
        for b in np.unique(which):
            m = which == b
            rows.append({"outcome": outcome, "bin": int(b), "predicted": float(p[m].mean()),
                         "actual": float(actual[m].mean()), "n": int(m.sum())})  # fmt: skip
    return pd.DataFrame(rows)


def binary_calibration(pred: np.ndarray, actual: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.unique(np.quantile(pred, np.linspace(0, 1, bins + 1)))
    which = np.clip(np.searchsorted(edges, pred, side="right") - 1, 0, max(len(edges) - 2, 0))
    rows = [{"predicted": float(pred[which == b].mean()), "actual": float(actual[which == b].mean()),
             "n": int((which == b).sum())} for b in np.unique(which)]  # fmt: skip
    return pd.DataFrame(rows)


def to_markdown_table(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    def fmt(v) -> str:
        return float_fmt.format(v) if isinstance(v, float) else str(v)

    header = "| " + " | ".join(map(str, df.columns)) + " |"
    divider = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([header, divider, *rows])
