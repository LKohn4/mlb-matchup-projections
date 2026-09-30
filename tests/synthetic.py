"""Synthetic, internally consistent Statcast-shaped data for offline tests.

Games have real structure: batting orders that cycle, starters who give way to relievers,
multi-pitch plate appearances, and outcomes driven by batter and pitcher skill, so every
part of the pipeline can be exercised without downloading anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TEAMS = ("NYY", "BOS", "LAD", "SF", "HOU", "ATL")
PITCH_TYPES = ("FF", "SI", "SL", "CU", "CH", "ST")
FIRST = ("Jake", "Luis", "Owen", "Mateo", "Cole", "Andre", "Kenji", "Tyler", "Marcus", "Rafael",
         "Nolan", "Isaiah", "Evan", "Diego", "Noah", "Caleb", "Hugo", "Ian", "Jonah", "Kai")
LAST = ("Rivera", "Chen", "Brooks", "Santana", "Walsh", "Okafor", "Moreno", "Park", "Hayes",
        "Duran", "Kim", "Foster", "Reyes", "Cruz", "Lopez", "Ward", "Hale", "Ito", "Nash", "Ruiz")


def _players(rng):
    batters, pitchers = {}, {}
    for t, team in enumerate(TEAMS):
        for j in range(12):
            pid = 600_000 + t * 100 + j
            batters[pid] = {
                "team": team,
                "stand": "S" if j == 2 else ("L" if j % 3 == 0 else "R"),
                "k": rng.uniform(0.14, 0.30), "bb": rng.uniform(0.05, 0.13),
                "power": rng.uniform(0.015, 0.06), "contact": rng.uniform(0.26, 0.36),
            }  # fmt: skip
        for j in range(8):
            pid = 650_000 + t * 100 + j
            mix = rng.dirichlet(np.ones(len(PITCH_TYPES)) * 2)
            pitchers[pid] = {
                "team": team, "throws": "L" if j % 4 == 0 else "R", "starter": j < 5,
                "k": rng.uniform(0.17, 0.32), "bb": rng.uniform(0.05, 0.11),
                "hr": rng.uniform(0.02, 0.045), "mix": mix,
            }  # fmt: skip
    return batters, pitchers


def make_statcast(n_games: int = 160, seed: int = 0, start: str = "2026-03-25",
                  end: str = "2026-09-27") -> pd.DataFrame:  # fmt: skip
    """Games shaped like pybaseball's output, including its nullable column types and the
    occasional batted ball with no tracked exit velocity."""
    rng = np.random.default_rng(seed)
    batters, pitchers = _players(rng)
    roster_b = {t: [p for p, v in batters.items() if v["team"] == t] for t in TEAMS}
    roster_p = {t: [p for p, v in pitchers.items() if v["team"] == t] for t in TEAMS}
    days = pd.date_range(start, end)
    dates = np.sort(rng.choice(days, n_games))
    rotation = {t: 0 for t in TEAMS}
    rows = []

    for g, date in enumerate(dates):
        home, away = rng.choice(TEAMS, 2, replace=False)
        game_pk = 800_000 + g
        lineups = {}
        for team in (away, home):
            order = roster_b[team][:9]
            if rng.random() < 0.3:  # occasional bench start
                order = order.copy()
                order[rng.integers(9)] = roster_b[team][9 + rng.integers(3)]
            lineups[team] = order
        starters = {}
        for team in (away, home):
            starters[team] = roster_p[team][rotation[team] % 5]
            rotation[team] += 1
        staff = {team: {"cur": starters[team], "bf": 0} for team in (away, home)}
        pos = {away: 0, home: 0}
        score = {away: 0, home: 0}
        ab = 0
        for inning in range(1, 10):
            for half, (bat, fld) in enumerate(((away, home), (home, away))):
                outs, bases = 0, 0
                while outs < 3:
                    st = staff[fld]
                    if st["cur"] == starters[fld] and st["bf"] >= rng.integers(20, 27):
                        st["cur"] = roster_p[fld][5 + rng.integers(3)]
                        st["bf"] = 0
                    pid, bid = st["cur"], lineups[bat][pos[bat]]
                    P, B = pitchers[pid], batters[bid]
                    stand = B["stand"] if B["stand"] != "S" else ("L" if P["throws"] == "R" else "R")
                    plat = 0.02 if stand != P["throws"] else -0.02
                    pk = np.clip((B["k"] + P["k"]) / 2 - plat, 0.05, 0.5)
                    pb = np.clip((B["bb"] + P["bb"]) / 2 + plat / 2, 0.02, 0.2)
                    ph = np.clip((B["power"] + P["hr"]) / 2 + plat / 3, 0.005, 0.1)
                    rest = 1 - pk - pb - ph
                    p_hit = B["contact"] + plat
                    probs = np.array([pk, pb, rest * p_hit * 0.68, rest * p_hit * 0.2,
                                      rest * p_hit * 0.02, ph, 0])  # fmt: skip
                    probs[6] = 1 - probs.sum()
                    outcome = rng.choice(7, p=probs)
                    event = ("strikeout", "walk", "single", "double", "triple", "home_run",
                             "field_out")[outcome]  # fmt: skip
                    if outcome == 1 and rng.random() < 0.08:
                        event = "hit_by_pitch"
                    if outcome == 6 and bases and rng.random() < 0.1:
                        event = "grounded_into_double_play"
                    if rng.random() < 0.01:
                        event = "intent_walk"
                    ab += 1
                    n = int(rng.integers(1, 7))
                    for k in range(1, n + 1):
                        last = k == n
                        desc = rng.choice(["ball", "called_strike", "foul", "swinging_strike"])
                        if last:
                            desc = {"strikeout": "swinging_strike", "walk": "ball",
                                    "intent_walk": "ball", "hit_by_pitch": "hit_by_pitch"
                                    }.get(event, "hit_into_play")  # fmt: skip
                        in_play = last and desc == "hit_into_play"
                        ev = 104 if outcome == 5 else (95 if outcome in (3, 4) else 88)
                        rows.append({
                            "game_date": date, "game_pk": game_pk, "game_type": "R",
                            "at_bat_number": ab, "pitch_number": k, "inning": inning,
                            "inning_topbot": "Top" if half == 0 else "Bot",
                            "home_team": home, "away_team": away, "batter": bid, "pitcher": pid,
                            "stand": stand, "p_throws": P["throws"],
                            "events": event if last else None, "description": desc,
                            "pitch_type": PITCH_TYPES[rng.choice(len(PITCH_TYPES), p=P["mix"])],
                            "launch_speed": rng.normal(ev, 8) if in_play else np.nan,
                            "post_bat_score": score[bat],
                            "player_name": f"{LAST[(pid // 7) % 20]}, {FIRST[pid % 20]}",
                        })  # fmt: skip
                    # crude scoring so post_bat_score is meaningful
                    if outcome == 5:
                        score[bat] += 1 + bin(bases).count("1")
                        bases = 0
                    elif outcome in (1, 2, 3, 4):
                        score[bat] += int(bases >= 4 and rng.random() < 0.5)
                        bases = min(7, (bases << (outcome - 1 if outcome > 1 else 1)) | 1)
                    else:
                        outs += 2 if event == "grounded_into_double_play" else 1
                    rows[-1]["post_bat_score"] = score[bat]
                    st["bf"] += 1
                    pos[bat] = (pos[bat] + 1) % 9
    df = pd.DataFrame(rows)
    in_play = df.index[df["description"].eq("hit_into_play")]
    untracked = rng.choice(in_play, size=max(1, len(in_play) // 50), replace=False)
    df.loc[untracked, "launch_speed"] = np.nan
    return df.convert_dtypes(convert_string=False)  # pybaseball returns nullable dtypes


def fake_names(raw: pd.DataFrame) -> pd.DataFrame:
    ids = pd.unique(pd.concat([raw["batter"], raw["pitcher"]]))
    names = [f"{FIRST[i % 20]} {LAST[(i // 3) % 20]}" for i in ids]
    names[0] = names[1]  # guarantee one ambiguous name for resolution tests
    return pd.DataFrame({"id": ids, "name": names})
