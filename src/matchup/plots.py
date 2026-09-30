"""Dashboard figures in one dark theme, matching the mlb-hit-probability project.

Every figure has a title, subtitle and a source/credit footer so it can be posted alone
or tiled into a dashboard.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

THEME = {"bg": "#0E1117", "panel": "#161B22", "text": "#E6EDF3", "muted": "#8B949E",
         "grid": "#30363D", "orange": "#F97316", "blue": "#38BDF8", "purple": "#A78BFA",
         "green": "#22C55E", "red": "#EF4444", "yellow": "#FACC15"}  # fmt: skip
SERIES = ["#F97316", "#38BDF8", "#A78BFA", "#22C55E", "#FACC15", "#F472B6"]
HEAT = LinearSegmentedColormap.from_list(
    "heat", ["#1E3A8A", "#2563EB", "#38BDF8", "#FDE047", "#F97316", "#DC2626"]
)
SOURCE = "Data: MLB Statcast via Baseball Savant & pybaseball"
FIGSIZE, DPI = (10, 6.25), 200


def avg_fmt(x: float) -> str:
    return f"{x:.3f}".lstrip("0") if 0 <= x < 1 else f"{x:.3f}"


@contextmanager
def _style():
    rc = {"figure.facecolor": THEME["bg"], "axes.facecolor": THEME["bg"],
          "savefig.facecolor": THEME["bg"], "axes.edgecolor": THEME["grid"],
          "axes.labelcolor": THEME["muted"], "axes.titlecolor": THEME["text"],
          "xtick.color": THEME["muted"], "ytick.color": THEME["muted"], "text.color": THEME["text"],
          "axes.grid": True, "grid.color": THEME["grid"], "grid.linewidth": 0.6,
          "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
          "font.size": 11, "axes.titlesize": 12.5, "legend.frameon": False,
          "legend.labelcolor": THEME["text"]}  # fmt: skip
    with plt.rc_context(rc):
        yield


def styled(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with _style():
            return func(*args, **kwargs)

    return wrapper


def _figure(title, subtitle, credit, nrows=1, ncols=1, figsize=FIGSIZE, **adjust):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    margins = {"top": 0.80, "bottom": 0.13, "left": 0.09, "right": 0.96, "wspace": 0.3, "hspace": 0.45}
    fig.subplots_adjust(**{**margins, **adjust})
    h = figsize[1]
    fig.text(0.04, 1 - 0.28 / h, title, fontsize=20, fontweight="bold", va="top")
    fig.text(0.04, 1 - 0.72 / h, subtitle, fontsize=11.5, va="top", color=THEME["muted"], wrap=True)
    fig.text(0.04, 0.025, SOURCE + (f"   |   {credit}" if credit else ""), fontsize=9,
             color=THEME["muted"])  # fmt: skip
    return fig, axes


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def _colorbar(fig, mappable, ax, label, fmt=avg_fmt):
    cb = fig.colorbar(mappable, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(label, color=THEME["muted"])
    cb.outline.set_edgecolor(THEME["grid"])
    cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fmt(v)))
    return cb


def _pct(ax, axis="y", digits=None):
    if digits is None:
        lo, hi = ax.get_ylim() if axis == "y" else ax.get_xlim()
        digits = 1 if hi - lo < 0.05 else 0
    f = plt.FuncFormatter(lambda v, _: f"{v:.{digits}%}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


# ====================================================================== season / model


@styled
def kpi_card(kpis: list[tuple[str, str, str]], path: Path, title: str, subtitle: str, credit: str):
    fig, axes = _figure(title, subtitle, credit, ncols=len(kpis), figsize=(12, 4.2), top=0.72,
                        bottom=0.14, left=0.03, right=0.97, wspace=0.08)  # fmt: skip
    for ax, (value, label, caption), color in zip(axes, kpis, SERIES, strict=False):
        ax.set_axis_off()
        ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0,rounding_size=0.06",
                                    transform=ax.transAxes, facecolor=THEME["panel"],
                                    edgecolor=THEME["grid"]))  # fmt: skip
        ax.text(0.5, 0.62, value, ha="center", va="center", fontsize=28, fontweight="bold",
                color=color, transform=ax.transAxes)  # fmt: skip
        ax.text(0.5, 0.33, label, ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.text(0.5, 0.16, caption, ha="center", va="center", fontsize=9, color=THEME["muted"],
                transform=ax.transAxes)  # fmt: skip
    return _save(fig, path)


@styled
def model_scorecard(scores: pd.DataFrame, path: Path, season: str, credit: str):
    order = ["League average", "Log5", "Random forest", "Gradient boosting", "Neural network"]
    s = scores.set_index("model").reindex([m for m in order if m in set(scores["model"])])
    fig, (a1, a2) = _figure(
        "Model Scorecard",
        f"Walk-forward backtest, {season}. Each PA predicted from earlier games only. Higher = better.",
        credit, ncols=2, left=0.17, wspace=0.12, top=0.78,
    )  # fmt: skip
    colors = [THEME["muted"], THEME["purple"], THEME["orange"], THEME["green"], THEME["blue"]]
    colors = [colors[order.index(m)] for m in s.index]
    y = np.arange(len(s))[::-1]
    for ax, col, title in ((a1, "skill_vs_league_pct", "Log-loss skill vs. league average"),
                           (a2, "brier_skill_pct", "Brier skill vs. league average")):  # fmt: skip
        vals = s[col].to_numpy()
        ax.barh(y, vals, color=colors, height=0.62)
        for yi, v in zip(y, vals, strict=True):
            ax.text(max(v, 0), yi, f" {v:+.2f}%", va="center", ha="left", fontsize=9)
        ax.axvline(0, color=THEME["text"], lw=1)
        lim = max(np.abs(vals).max() * 1.45, 0.1)
        ax.set_xlim(min(vals.min() * 1.2, -lim * 0.1), lim)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.1f}%"))
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
    a1.set_yticks(y, s.index)
    a2.set_yticks(y, [""] * len(s))
    return _save(fig, path)


@styled
def calibration_by_outcome(cal: pd.DataFrame, path: Path, model: str, season: str, credit: str):
    outcomes = list(dict.fromkeys(cal["outcome"]))
    fig, axes = _figure(
        "Are the Probabilities Honest?",
        f"{model}: predicted vs. actual rate for each outcome, in 10 equal-size groups of test "
        f"plate appearances, {season}.", credit, nrows=2, ncols=4, figsize=(12, 7), left=0.06,
        top=0.82, bottom=0.1, wspace=0.28, hspace=0.45,
    )  # fmt: skip
    for ax, outcome, color in zip(axes.flat, outcomes, SERIES * 2, strict=False):
        d = cal.loc[cal["outcome"] == outcome]
        hi = max(d["predicted"].max(), d["actual"].max()) * 1.1
        ax.plot([0, hi], [0, hi], color=THEME["muted"], ls="--", lw=1)
        ax.plot(d["predicted"], d["actual"], marker="o", ms=4, lw=2, color=color)
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_title(outcome, loc="left")
        ax.locator_params(nbins=4)
        _pct(ax, "x")
        _pct(ax, "y")
        ax.tick_params(labelsize=8)
    for ax in list(axes.flat)[len(outcomes):]:
        ax.set_axis_off()
        ax.text(0.05, 0.6, "On the dashed line =\nprobabilities match\nreality", fontsize=11,
                color=THEME["muted"], transform=ax.transAxes)  # fmt: skip
    return _save(fig, path)


@styled
def ablation_chart(abl: pd.DataFrame, path: Path, season: str, credit: str):
    d = abl.sort_values("loss_increase_pct")
    fig, ax = _figure(
        "Which Variables Actually Matter?",
        f"How much worse predictions get when each variable group is removed, {season}. Bigger "
        "bar = more important. Values within about ±0.05% are run-to-run noise.", credit, left=0.27,
    )  # fmt: skip
    colors = [THEME["orange"] if v > 0.02 else THEME["muted"] for v in d["loss_increase_pct"]]
    ax.barh(d["group"], d["loss_increase_pct"], color=colors)
    for yi, v in enumerate(d["loss_increase_pct"]):
        ax.text(max(v, 0), yi, f" {v:+.3f}%", va="center", ha="left", fontsize=9)
    ax.axvline(0, color=THEME["text"], lw=1)
    lo, hi = d["loss_increase_pct"].min(), d["loss_increase_pct"].max()
    ax.set_xlim(min(lo * 1.1, 0), max(hi, 0) + (hi - min(lo, 0)) * 0.25 + 1e-3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.2f}%"))
    ax.set_xlabel("Increase in log loss when removed (% of league baseline)")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


@styled
def tto_chart(tto: pd.DataFrame, path: Path, season: str, credit: str):
    fig, (a1, a2) = _figure(
        "The Times-Through-the-Order Penalty",
        f"League results by how many times the pitcher has faced the lineup, {season}",
        credit, ncols=2, wspace=0.25,
    )  # fmt: skip
    x = [f"{int(t)}{'st' if t == 1 else 'nd' if t == 2 else 'rd' if t == 3 else 'th+'}" for t in tto["tto"]]
    a1.bar(x, tto["k_rate"], color=THEME["blue"])
    a2.bar(x, tto["woba"], color=THEME["orange"])
    for ax, col, fmt in ((a1, "k_rate", "{:.1%}"), (a2, "woba", None)):
        for xi, v in enumerate(tto[col]):
            ax.text(xi, v, fmt.format(v) if fmt else avg_fmt(v), ha="center", va="bottom", fontsize=10)
        ax.grid(axis="x", visible=False)
        ax.set_xlabel("Time through the order")
    a1.set_title("Strikeout rate", loc="left")
    a2.set_title("wOBA allowed", loc="left")
    _pct(a1)
    a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: avg_fmt(v)))
    return _save(fig, path)


@styled
def platoon_chart(pl: pd.DataFrame, path: Path, season: str, credit: str):
    fig, ax = _figure(
        "Platoon Splits",
        f"League wOBA by batter side and pitcher hand, {season}. Batters do better against "
        "opposite-handed pitchers.", credit,
    )  # fmt: skip
    groups = ["L", "R"]
    w = 0.35
    for j, (hand, color) in enumerate((("L", THEME["purple"]), ("R", THEME["orange"]))):
        vals = [pl.loc[(pl["stand"] == g) & (pl["p_throws"] == hand), "woba"].mean() for g in groups]
        xs = np.arange(2) + (j - 0.5) * w
        ax.bar(xs, vals, width=w, color=color, label=f"vs {'LHP' if hand == 'L' else 'RHP'}")
        for x, v in zip(xs, vals, strict=True):
            ax.text(x, v, avg_fmt(v), ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(2), ["Left-handed batters", "Right-handed batters"])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: avg_fmt(v)))
    ax.set_ylabel("wOBA")
    ax.legend()
    ax.grid(axis="x", visible=False)
    return _save(fig, path)


@styled
def pitch_family_chart(fam: pd.DataFrame, path: Path, season: str, credit: str):
    fig, (a1, a2) = _figure(
        "How Batters Handle Each Pitch Family",
        f"League whiff rate per swing and wOBA on plate appearances ending on each family, "
        f"{season}", credit, ncols=2, wspace=0.25,
    )  # fmt: skip
    fams = list(dict.fromkeys(fam["family"]))
    w = 0.35
    for j, (side, color) in enumerate((("L", THEME["purple"]), ("R", THEME["orange"]))):
        d = fam.loc[fam["stand"] == side].set_index("family").reindex(fams)
        xs = np.arange(len(fams)) + (j - 0.5) * w
        a1.bar(xs, d["whiff_rate"], width=w, color=color, label=f"{side}HB")
        a2.bar(xs, d["woba"], width=w, color=color, label=f"{side}HB")
    for ax, title in ((a1, "Whiff rate"), (a2, "wOBA")):
        ax.set_xticks(range(len(fams)), [f.title() for f in fams])
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
        ax.legend(ncol=2, loc="upper right")
    _pct(a1)
    a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: avg_fmt(v)))
    return _save(fig, path)


@styled
def hot_cold_chart(hc: pd.DataFrame, path: Path, season: str, credit: str):
    fig, ax = _figure(
        "Are Hot Streaks Real?",
        f"Batters grouped by how far their last 14 days ran above/below their season on-base "
        f"rate, and how they did in their next plate appearance, {season}.", credit,
    )  # fmt: skip
    lim = float(np.nanmax(np.abs(hc["recent_delta"]))) * 1.1
    ax.plot([-lim, lim], [-lim, lim], color=THEME["muted"], ls="--", lw=1,
            label="If streaks fully continued")  # fmt: skip
    ax.plot(hc["recent_delta"], hc["next_delta"], marker="o", lw=2.5, color=THEME["orange"],
            label="What actually happened")  # fmt: skip
    ax.axhline(0, color=THEME["grid"], lw=1)
    ax.axvline(0, color=THEME["grid"], lw=1)
    ax.set_xlabel("Recent on-base rate minus season rate (cold ← → hot)")
    ax.set_ylabel("Next-PA on-base rate minus season rate")
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.3f}"))
    ax.legend(loc="upper left")
    return _save(fig, path)


@styled
def park_factors_chart(parks: pd.DataFrame, path: Path, season: str, credit: str):
    d = parks.sort_values("hr_factor")
    fig, ax = _figure(
        "Park Factors",
        f"Home runs and hits per plate appearance at each park relative to league average "
        f"(1.00 = neutral), shrunk for sample size, {season}", credit, left=0.1,
    )  # fmt: skip
    y = np.arange(len(d))
    ax.barh(y, d["hr_factor"] - 1, left=1, color=[THEME["orange"] if v >= 1 else THEME["blue"]
                                                   for v in d["hr_factor"]], height=0.6,
            label="Home run factor")  # fmt: skip
    ax.scatter(d["hit_factor"], y, color=THEME["text"], s=18, zorder=3, label="Hit factor")
    ax.axvline(1, color=THEME["text"], lw=1)
    ax.set_yticks(y, d["park"], fontsize=8)
    ax.set_xlabel("Factor (1.00 = league average)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


@styled
def situational_chart(sit: pd.DataFrame, path: Path, season: str, credit: str):
    fig, axes = _figure(
        "Home vs. Away, Day vs. Night",
        f"League wOBA by game situation, {season}. Small gaps mean small effects.", credit,
        ncols=2, wspace=0.25,
    )  # fmt: skip
    for ax, (kind, title) in zip(axes, (("venue", "Home vs. away"), ("time", "Day vs. night")),
                                 strict=True):  # fmt: skip
        d = sit.loc[sit["kind"] == kind]
        if d.empty:
            ax.set_axis_off()
            ax.set_title(title, loc="left")
            ax.text(0.5, 0.5, "Not available\n(MLB schedule couldn't be reached)", ha="center",
                    va="center", color=THEME["muted"], transform=ax.transAxes)  # fmt: skip
            continue
        ax.bar(d["label"], d["woba"], color=[THEME["orange"], THEME["blue"]][: len(d)], width=0.55)
        for xi, (v, n) in enumerate(zip(d["woba"], d["n"], strict=True)):
            ax.text(xi, v, f"{avg_fmt(v)}\n{n:,} PA", ha="center", va="bottom", fontsize=9)
        lo = d["woba"].min() * 0.9
        ax.set_ylim(lo, d["woba"].max() * 1.06)
        ax.set_title(title, loc="left")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: avg_fmt(v)))
        ax.grid(axis="x", visible=False)
    return _save(fig, path)


@styled
def game_calibration_chart(cal: pd.DataFrame, path: Path, n_games: int, season: str, credit: str):
    fig, axes = _figure(
        "Full-Game Projections vs. Reality",
        f"{n_games} real games from the final backtest window, simulated before they were "
        f"played. Predicted vs. actual share of starters with each result, {season}.", credit,
        ncols=2, wspace=0.25, top=0.76,
    )  # fmt: skip
    for ax, (event, color) in zip(axes, (("1+ hit", THEME["orange"]), ("1+ strikeout", THEME["blue"])),
                                  strict=True):  # fmt: skip
        d = cal.loc[cal["event"] == event]
        lo = min(d["predicted"].min(), d["actual"].min()) * 0.9
        hi = max(d["predicted"].max(), d["actual"].max()) * 1.05
        ax.plot([lo, hi], [lo, hi], color=THEME["muted"], ls="--", lw=1)
        ax.plot(d["predicted"], d["actual"], marker="o", lw=2.5, color=color)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(f"Chance of {event}", loc="left")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        _pct(ax, "x")
        _pct(ax, "y")
    return _save(fig, path)


# ====================================================================== single game


@styled
def game_preview(summary: dict, away: str, home: str, info: str, starters: tuple[str, str],
                 score: np.ndarray, path: Path, credit: str):  # fmt: skip
    fig = plt.figure(figsize=FIGSIZE)
    fig.text(0.04, 0.955, f"{away} at {home}", fontsize=24, fontweight="bold", va="top")
    fig.text(0.04, 0.875, info, fontsize=11.5, color=THEME["muted"], va="top")
    fig.text(0.04, 0.025, SOURCE + (f"   |   {credit}" if credit else ""), fontsize=9,
             color=THEME["muted"])  # fmt: skip

    home_p = summary["home_win_prob"]
    ax = fig.add_axes([0.04, 0.64, 0.92, 0.1])
    ax.set_axis_off()
    ax.barh(0, 1 - home_p, color=THEME["blue"], height=0.8)
    ax.barh(0, home_p, left=1 - home_p, color=THEME["orange"], height=0.8)
    ax.text(0.01, 0, f"{away}  {1 - home_p:.0%}", va="center", fontsize=15, fontweight="bold",
            color=THEME["bg"])  # fmt: skip
    ax.text(0.99, 0, f"{home_p:.0%}  {home}", va="center", ha="right", fontsize=15,
            fontweight="bold", color=THEME["bg"])  # fmt: skip
    ax.set_xlim(0, 1)
    fig.text(0.04, 0.765, "Win probability", fontsize=12, color=THEME["muted"])

    for x, team, runs, sp, color in ((0.04, away, summary["away_runs"], starters[0], THEME["blue"]),
                                     (0.26, home, summary["home_runs"], starters[1], THEME["orange"])):  # fmt: skip
        fig.text(x, 0.52, f"{runs:.1f}", fontsize=34, fontweight="bold", color=color)
        fig.text(x, 0.47, f"{team} projected runs", fontsize=10.5, color=THEME["muted"])
        fig.text(x, 0.37, "Starter", fontsize=10, color=THEME["muted"])
        fig.text(x, 0.32, sp, fontsize=12.5)
    fig.text(0.04, 0.2, f"Total runs {summary['total_runs']:.1f}   ·   extra innings "
             f"{summary['p_extra_innings']:.0%}   ·   {summary['sims']:,} simulations", fontsize=10.5,
             color=THEME["muted"])  # fmt: skip

    ax2 = fig.add_axes([0.53, 0.14, 0.43, 0.42])
    runs = np.arange(15)
    for k, (col, team, color) in enumerate(((0, away, THEME["blue"]), (1, home, THEME["orange"]))):
        share = np.bincount(np.minimum(score[:, col], 14), minlength=15) / len(score)
        ax2.bar(runs + (k - 0.5) * 0.4, share, width=0.4, color=color, label=team)
    ax2.set_title("Runs scored distribution", loc="left", fontsize=11)
    ax2.set_xlim(-0.5, 14.5)
    ax2.legend()
    _pct(ax2)
    return _save(fig, path)


@styled
def matchup_heatmaps(mats: list[pd.DataFrame], teams: list[str], path: Path, subtitle: str,
                     credit: str, title: str = "Batter vs. Pitcher Matchups",
                     label: str = "Projected wOBA per PA"):  # fmt: skip
    fig, axes = _figure(title, subtitle, credit, ncols=2, figsize=(14, 7.5), left=0.12,
                        wspace=0.55, bottom=0.2, right=0.95)  # fmt: skip
    vals = np.concatenate([m.to_numpy().ravel() for m in mats])
    vmin, vmax = np.nanpercentile(vals, 3), np.nanpercentile(vals, 97)
    for ax, mat, team in zip(axes, mats, teams, strict=True):
        im = ax.imshow(mat.to_numpy(), cmap=HEAT, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(mat.shape[1]), mat.columns, fontsize=8, rotation=35, ha="right")
        ax.set_yticks(range(mat.shape[0]), mat.index, fontsize=8.5)
        ax.grid(False)
        ax.set_title(f"{team} hitters", loc="left")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.iat[i, j]
                ax.text(j, i, avg_fmt(v), ha="center", va="center", fontsize=8,
                        color=THEME["bg"] if (v - vmin) / max(vmax - vmin, 1e-9) > 0.45 else THEME["text"])  # fmt: skip
    _colorbar(fig, im, axes[-1], label)
    return _save(fig, path)


@styled
def projected_lines_table(lines: pd.DataFrame, teams: list[str], path: Path, subtitle: str,
                          credit: str):  # fmt: skip
    cols = ["PA", "H", "HR", "K", "BB", "TB", "1+ H", "xwOBA"]
    fig, axes = _figure("Projected Batting Lines", subtitle, credit, nrows=2, figsize=(12, 8.2),
                        top=0.87, bottom=0.05, left=0.03, right=0.97, hspace=0.05)  # fmt: skip
    for ax, t, team in zip(axes, (0, 1), teams, strict=True):
        ax.set_axis_off()
        d = lines.loc[lines["team"] == t].sort_values("slot")
        cells = [[f"{r.slot}. {r.name}", f"{r.pa:.1f}", f"{r.h:.2f}", f"{r.hr:.2f}", f"{r.k:.2f}",
                  f"{r.bb:.2f}", f"{r.tb:.2f}", f"{r.p_1plus_h:.0%}", avg_fmt(r.xwoba)]
                 for r in d.itertuples()]  # fmt: skip
        table = ax.table(cellText=cells, colLabels=[f"{team} lineup", *cols], loc="center",
                         cellLoc="center", colWidths=[0.28] + [0.09] * len(cols))  # fmt: skip
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.45)
        best = d["xwoba"].idxmax() if len(d) else None
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor(THEME["grid"])
            cell.set_facecolor(THEME["panel"] if row == 0 else THEME["bg"])
            cell.get_text().set_color(THEME["text"] if row else THEME["muted"])
            if col == 0 and row:
                cell.get_text().set_ha("left")
                cell._loc = "left"
            if row and best is not None and d.index[row - 1] == best:
                cell.get_text().set_color(THEME["orange"])
    return _save(fig, path)
