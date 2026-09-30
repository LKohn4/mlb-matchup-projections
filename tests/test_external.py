import pandas as pd

from matchup.external import (
    day_night_map,
    load_player_names,
    parse_people,
    parse_schedule,
    pitcher_names_from_statcast,
)

SCHEDULE = {"dates": [{"date": "2026-04-01", "games": [
    {"gamePk": 1, "dayNight": "day", "venue": {"name": "Wrigley Field"}},
    {"gamePk": 2, "dayNight": "night", "venue": {"name": "Fenway Park"}},
    {"gamePk": 3, "venue": {}}]}]}


def test_parse_schedule_and_day_night():
    s = parse_schedule(SCHEDULE)
    assert s["game_pk"].tolist() == [1, 2, 3]
    m = day_night_map(s)
    assert m.to_dict() == {1: 0.0, 2: 1.0}  # game 3 unknown -> left out


def test_parse_people():
    payload = {"people": [{"id": 5, "fullName": "Test Player"}, {"id": 6}]}
    assert parse_people(payload) == {5: "Test Player"}


def test_pitcher_names_fallback(raw):
    names = pitcher_names_from_statcast(raw)
    pid = int(raw["pitcher"].iloc[0])
    last, first = raw["player_name"].iloc[0].split(", ")
    assert names[pid] == f"{first} {last}"


def test_offline_names_use_cache_then_fallback(raw, tmp_path):
    cache = tmp_path / "players.csv"
    pd.DataFrame({"id": [600000], "name": ["Cached Name"]}).to_csv(cache, index=False)
    ids = [600000, int(raw["pitcher"].iloc[0])]
    table = load_player_names(ids, cache, raw, fetch=False)
    got = dict(zip(table["id"], table["name"], strict=True))
    assert got[600000] == "Cached Name" and ids[1] in got


def test_postponed_game_listed_twice():
    """The Stats API lists a postponed game on its original and make-up dates."""
    payload = {"dates": [
        {"date": "2026-04-10", "games": [
            {"gamePk": 7, "dayNight": "night", "status": {"detailedState": "Postponed"}}]},
        {"date": "2026-04-11", "games": [
            {"gamePk": 7, "dayNight": "day", "status": {"detailedState": "Final"}}]},
    ]}
    m = day_night_map(parse_schedule(payload))
    assert m.index.is_unique and m.to_dict() == {7: 0.0}  # the make-up day game


def test_old_cache_without_status_still_unique():
    old = pd.DataFrame({"game_pk": [7, 7, 8], "day_night": ["night", "day", "night"]})
    m = day_night_map(old)
    assert m.index.is_unique and m.to_dict() == {7: 0.0, 8: 1.0}
