import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from elo_ratings import SURFACES

ALLOWED_TOURS = ("ATP", "WTA")
# 1-3 uppercase letters covers every code actually seen in draw sheets: Q (qualifier),
# W/WC (wildcard), L (lucky loser), PR (protected ranking), etc.
STATUS_RE = re.compile(r"^[A-Z]{1,3}$")

REQUIRED_TOP_LEVEL_FIELDS = ("tournament", "year", "tour", "surface", "start_date", "players")


class BracketValidationError(ValueError):
    pass


@dataclass(frozen=True)
class PlayerEntry:
    seed: Optional[int]
    name: str
    status: Optional[str]
    bye: bool = False
    # slot number (1..N) in the full bracket, e.g. from a PDF-extracted draw sheet where some
    # slots are omitted because they're the "phantom" opponent of a bye. Optional - a hand-written
    # bracket YAML can omit it entirely and rely on list order instead. When present on every
    # player, it's used as the authoritative draw order instead of trusting the YAML's list order.
    position: Optional[int] = None


@dataclass(frozen=True)
class Bracket:
    tournament: str
    year: int
    tour: str
    surface: str
    start_date: pd.Timestamp
    players: list
    source_path: Path


def _validate_player(entry, index, errors):
    prefix = f"players[{index}]"
    if not isinstance(entry, dict):
        errors.append(f"{prefix}: expected a mapping with seed/name/status, got {type(entry).__name__}")
        return None

    name = entry.get("name")
    if not name or not isinstance(name, str) or not name.strip():
        errors.append(f"{prefix}: missing or empty required field 'name'")
        name = None

    seed = entry.get("seed")
    if seed is not None and not isinstance(seed, int):
        errors.append(f"{prefix}: 'seed' must be an integer or null, got {seed!r}")
        seed = None

    status = entry.get("status")
    if status is not None and (not isinstance(status, str) or not STATUS_RE.match(status)):
        errors.append(f"{prefix}: 'status' must be 1-3 uppercase letters (e.g. Q, WC, L, PR) or null, got {status!r}")
        status = None

    bye = entry.get("bye", False)
    if bye is None:
        bye = False
    if not isinstance(bye, bool):
        errors.append(f"{prefix}: 'bye' must be true/false, got {bye!r}")
        bye = False

    position = entry.get("position")
    if position is not None and not isinstance(position, int):
        errors.append(f"{prefix}: 'position' must be an integer or null, got {position!r}")
        position = None

    if name is None:
        return None
    return PlayerEntry(seed=seed, name=name.strip(), status=status, bye=bye, position=position)


def _validate(data):
    errors = []
    if not isinstance(data, dict):
        return [], ["Bracket file must contain a YAML mapping at the top level"]

    for field in REQUIRED_TOP_LEVEL_FIELDS:
        if field not in data or data[field] in (None, ""):
            errors.append(f"missing required field: '{field}'")

    year = data.get("year")
    if year is not None and not isinstance(year, int):
        errors.append(f"'year' must be an integer, got {year!r}")

    tour = data.get("tour")
    if tour is not None and (not isinstance(tour, str) or tour.upper() not in ALLOWED_TOURS):
        errors.append(f"'tour' must be one of {ALLOWED_TOURS}, got {tour!r}")

    surface = data.get("surface")
    if surface is not None and (not isinstance(surface, str) or surface not in SURFACES):
        errors.append(f"'surface' must be one of {list(SURFACES)}, got {surface!r}")

    start_date = data.get("start_date")
    parsed_start_date = None
    if start_date is not None:
        try:
            parsed_start_date = pd.Timestamp(start_date)
        except (ValueError, TypeError):
            errors.append(f"'start_date' must be a parseable date (YYYY-MM-DD), got {start_date!r}")

    players_raw = data.get("players")
    players = []
    if players_raw is not None:
        if not isinstance(players_raw, list):
            errors.append(f"'players' must be a list, got {type(players_raw).__name__}")
        else:
            for index, entry in enumerate(players_raw):
                player = _validate_player(entry, index, errors)
                if player is not None:
                    players.append(player)
            if not players_raw:
                errors.append("'players' must not be empty")

            seeds = [p.seed for p in players if p.seed is not None]
            duplicate_seeds = {seed for seed in seeds if seeds.count(seed) > 1}
            if duplicate_seeds:
                errors.append(f"duplicate seed value(s): {sorted(duplicate_seeds)}")

            positions = [p.position for p in players if p.position is not None]
            duplicate_positions = {pos for pos in positions if positions.count(pos) > 1}
            if duplicate_positions:
                errors.append(f"duplicate position value(s): {sorted(duplicate_positions)}")

    return {
        "tournament": data.get("tournament"),
        "year": year,
        "tour": tour.upper() if isinstance(tour, str) else tour,
        "surface": surface,
        "start_date": parsed_start_date,
        "players": players,
    }, errors


def load_bracket_yaml(path):
    path = Path(path)
    if not path.exists():
        raise BracketValidationError(f"Bracket file not found: {path}")

    with open(path, encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise BracketValidationError(f"{path}: invalid YAML — {e}") from e

    parsed, errors = _validate(data)
    if errors:
        formatted = "\n".join(f"  - {error}" for error in errors)
        raise BracketValidationError(f"{path}: bracket schema validation failed:\n{formatted}")

    return Bracket(
        tournament=parsed["tournament"],
        year=parsed["year"],
        tour=parsed["tour"],
        surface=parsed["surface"],
        start_date=parsed["start_date"],
        players=parsed["players"],
        source_path=path,
    )
