"""
pokemon_data.py — comprehensive Pokémon name list + hint-matching utilities.

Poketwo gives hints in the form:
    "The pokémon is **C\\_\\_ \\_ \\_z\\_\\_ \\_**."
where letters and blanks are mixed. This module parses those hints and finds
the best-matching Pokémon name.
"""

import re
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple
from difflib import SequenceMatcher

logger = logging.getLogger("pokemon_data")

# ── Load names from the trained-model index if available ──────────────────────
_BASE_DIR = Path(__file__).resolve().parent.parent
_INDEX_PATH = _BASE_DIR / "model" / "index_to_pokemon.json"

def _load_names_from_index() -> List[str]:
    """Pull canonical names from the model's index map."""
    if _INDEX_PATH.exists():
        with open(_INDEX_PATH, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return list(raw.values())
    return []


# ── Hardcoded master list (fallback + extended with every known Pokémon) ──────
# This list covers Gen I – IX base forms + common variants that Poketwo uses.
# Underscores have been replaced with spaces for matching.
ALL_POKEMON: List[str] = sorted(set(
    n.replace("_", " ") for n in _load_names_from_index()
)) if _INDEX_PATH.exists() else []

# If the index file wasn't available at import time, fall back to a minimal
# hard-coded list of Gen I-IX base names (abbreviated for size — the full model
# mapping is the canonical source).
if not ALL_POKEMON:
    ALL_POKEMON = sorted([
        "Bulbasaur", "Ivysaur", "Venusaur", "Charmander", "Charmeleon",
        "Charizard", "Squirtle", "Wartortle", "Blastoise", "Caterpie",
        "Metapod", "Butterfree", "Weedle", "Kakuna", "Beedrill",
        "Pidgey", "Pidgeotto", "Pidgeot", "Rattata", "Raticate",
        "Spearow", "Fearow", "Ekans", "Arbok", "Pikachu",
        "Raichu", "Sandshrew", "Sandslash", "Nidoran♀", "Nidorina",
        "Nidoqueen", "Nidoran♂", "Nidorino", "Nidoking", "Clefairy",
        "Clefable", "Vulpix", "Ninetales", "Jigglypuff", "Wigglytuff",
        "Zubat", "Golbat", "Oddish", "Gloom", "Vileplume",
        "Paras", "Parasect", "Venonat", "Venomoth", "Diglett",
        "Dugtrio", "Meowth", "Persian", "Psyduck", "Golduck",
        "Mankey", "Primeape", "Growlithe", "Arcanine", "Poliwag",
        "Poliwhirl", "Poliwrath", "Abra", "Kadabra", "Alakazam",
        "Machop", "Machoke", "Machamp", "Bellsprout", "Weepinbell",
        "Victreebel", "Tentacool", "Tentacruel", "Geodude", "Graveler",
        "Golem", "Ponyta", "Rapidash", "Slowpoke", "Slowbro",
        "Magnemite", "Magneton", "Farfetch\u2019d", "Doduo", "Dodrio",
        "Seel", "Dewgong", "Grimer", "Muk", "Shellder",
        "Cloyster", "Gastly", "Haunter", "Gengar", "Onix",
        "Drowzee", "Hypno", "Krabby", "Kingler", "Voltorb",
        "Electrode", "Exeggcute", "Exeggutor", "Cubone", "Marowak",
        "Hitmonlee", "Hitmonchan", "Lickitung", "Koffing", "Weezing",
        "Rhyhorn", "Rhydon", "Chansey", "Tangela", "Kangaskhan",
        "Horsea", "Seadra", "Goldeen", "Seaking", "Staryu",
        "Starmie", "Mr. Mime", "Scyther", "Jynx", "Electabuzz",
        "Magmar", "Pinsir", "Tauros", "Magikarp", "Gyarados",
        "Lapras", "Ditto", "Eevee", "Vaporeon", "Jolteon",
        "Flareon", "Porygon", "Omanyte", "Omastar", "Kabuto",
        "Kabutops", "Aerodactyl", "Snorlax", "Articuno", "Zapdos",
        "Moltres", "Dratini", "Dragonair", "Dragonite", "Mewtwo", "Mew",
    ])

# Build a lowercased lookup set for O(1) membership checks
_POKEMON_LOWER = {p.lower(): p for p in ALL_POKEMON}


# ── Hint Parsing ──────────────────────────────────────────────────────────────

def parse_hint(hint_text: str) -> Optional[str]:
    """Extract the hint pattern from a Pokétwo hint message.

    Poketwo sends hints like:
        ``The pokémon is **C\\_\\_ \\_ \\_z\\_\\_ \\_**.``

    We strip markdown, unescape, and return a clean pattern string where
    unknown letters are represented by ``_`` and known letters are kept.

    Returns ``None`` if parsing fails.
    """
    # Try to find content between ** **
    match = re.search(r"\*\*(.+?)\*\*", hint_text)
    if not match:
        return None

    raw = match.group(1).strip().rstrip(".")

    # Remove backslash escaping
    raw = raw.replace("\\_", "_")
    raw = raw.replace("\\", "")

    # Normalise whitespace
    pattern = raw.strip()
    return pattern if pattern else None


def _hint_matches(pattern: str, name: str) -> bool:
    """Check if a Pokémon name matches a parsed hint pattern character-by-character.

    Rules:
    * ``_`` in the pattern matches any single character in the name.
    * Literal characters must match (case-insensitive).
    * Lengths must be equal.
    """
    if len(pattern) != len(name):
        return False
    for pc, nc in zip(pattern.lower(), name.lower()):
        if pc == "_":
            continue  # wildcard
        if pc != nc:
            return False
    return True


def match_from_hint(pattern: str) -> List[str]:
    """Return all Pokémon names that match *pattern* exactly."""
    return [name for name in ALL_POKEMON if _hint_matches(pattern, name)]


def get_best_hint_match(hint_text: str) -> Optional[str]:
    """High-level helper: parse hint → match → return best result or None."""
    pattern = parse_hint(hint_text)
    if pattern is None:
        logger.debug("Could not parse hint from: %s", hint_text)
        return None

    matches = match_from_hint(pattern)
    logger.debug("Hint pattern '%s' matched %d names: %s", pattern, len(matches), matches[:10])

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # If multiple matches, pick the shortest (base-form bias)
        matches.sort(key=len)
        return matches[0]
    return None


def fuzzy_match(query: str, threshold: float = 0.6) -> Optional[str]:
    """Fuzzy-search *query* against ALL_POKEMON using SequenceMatcher.

    Returns the best match above *threshold*, or ``None``.
    """
    best_name: Optional[str] = None
    best_ratio: float = 0.0
    query_lower = query.lower()

    for name in ALL_POKEMON:
        ratio = SequenceMatcher(None, query_lower, name.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_name = name

    if best_ratio >= threshold and best_name is not None:
        return best_name
    return None
