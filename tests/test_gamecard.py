import pytest
import yaml

from matchup.gamecard import CardError, card_to_yaml, generate_card, load_card, parse_card
from matchup.players import ResolutionError
from matchup.project import resolve_card


def test_generated_card_roundtrips(season, directory, tmp_path):
    card = generate_card(season.pa_all, directory, "LAD", "NYY", "2026-10-02", "night")
    assert len(card.away.lineup) == len(card.home.lineup) == 9
    path = tmp_path / "card.yaml"
    path.write_text(card_to_yaml(card), encoding="utf-8")
    again = load_card(path)
    assert again.away.lineup == card.away.lineup and again.park == "NYY"
    teams, _ = resolve_card(again, directory)
    assert all(len(set(t.lineup)) == 9 for t in teams)
    assert teams[0].staff[1] is not None


def test_card_errors_are_collected():
    data = {"game": {"date": "not-a-date", "time": "evening"},
            "away": {"team": "LAD", "lineup": ["A"] * 8, "pitchers": {}},
            "home": {"team": "NYY", "lineup": ["B"] * 9, "pitchers": {"starter": "X",
                     "relievers": ["a", "b", "c", "d"]}}}
    with pytest.raises(CardError) as exc:
        parse_card(data)
    message = str(exc.value)
    for fragment in ("date", "time", "exactly 9", "starter", "at most 3"):
        assert fragment in message


def test_invalid_yaml(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("game: [unclosed", encoding="utf-8")
    with pytest.raises(CardError, match="valid YAML"):
        load_card(bad)


def test_resolve_by_name_id_and_team(directory):
    p = directory.players.iloc[5]
    assert directory.resolve(p["name"] if p["name"] not in directory.players.loc[
        directory.players.index != p["id"], "name"].values else f"ID {p['id']}") == p["id"]
    assert directory.resolve(f"ID {p['id']}") == p["id"]
    assert directory.resolve(int(p["id"])) == p["id"]
    assert directory.resolve({"name": "whatever", "id": int(p["id"])}) == p["id"]


def test_ambiguous_and_fuzzy_names(directory):
    dup = directory.players["name"].value_counts()
    shared = dup[dup > 1].index[0]
    with pytest.raises(ResolutionError, match="several players"):
        directory.resolve(shared)
    name = directory.players["name"].iloc[20]
    typo = name[:-1] + ("x" if name[-1] != "x" else "y")
    with pytest.raises(ResolutionError, match="Did you mean"):
        directory.resolve(typo)


def test_switch_hitter_takes_platoon_side(directory):
    switch = directory.players.loc[directory.players["stands"].fillna("").str.len() > 1, "id"].iloc[0]
    assert directory.stand_vs(switch, "R") == "L"
    assert directory.stand_vs(switch, "L") == "R"


def test_template_card_is_valid_yaml():
    from pathlib import Path

    template = Path(__file__).resolve().parents[1] / "game_cards" / "TEMPLATE.yaml"
    data = yaml.safe_load(template.read_text(encoding="utf-8"))
    assert parse_card(data).home.team
