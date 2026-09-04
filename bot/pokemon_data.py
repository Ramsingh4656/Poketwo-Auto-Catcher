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
import unicodedata
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

_AUTHORITATIVE_LABEL_COUNT = 936

_FORM_PREFIXES = {
    "alola": "alolan",
    "galar": "galarian",
    "hisui": "hisuian",
    "paldea": "paldean",
    "gmax": "gigantamax",
    "mega": "mega",
}


def _compact(value: str) -> str:
    """Normalize labels and hints to comparable alphanumeric text.

    Pokétwo may use hyphens, underscores, apostrophes, accents, or gender
    symbols inconsistently. Gender symbols are intentionally mapped to the
    same ``f``/``m`` suffix used by the authoritative ONNX labels.
    """
    value = value.replace("\\_", "_").replace("♀", "f").replace("♂", "m")
    value = unicodedata.normalize("NFKD", value)
    return "".join(ch.lower() for ch in value if ch.isalnum())


def _label_variants(name: str) -> set[str]:
    """Return canonical and safe human-form variants for one model label."""
    raw = name.replace("_", "-").strip().lower()
    variants = {_compact(name)}
    parts = [part for part in raw.split("-") if part]
    if len(parts) < 2:
        return variants

    # The metadata uses suffix forms such as articuno-galar and
    # charizard-mega-x, while Pokétwo hints commonly say Galarian Articuno or
    # Mega Charizard X. Add only known form reorderings.
    form = parts[-1]
    if form in _FORM_PREFIXES:
        variants.add(_compact(f"{_FORM_PREFIXES[form]} {' '.join(parts[:-1])}"))
    if len(parts) >= 3 and parts[-2] == "mega":
        variants.add(_compact(f"mega {' '.join(parts[:-2])} {parts[-1]}"))
    if form == "gmax":
        variants.add(_compact(f"gigantamax {' '.join(parts[:-1])}"))
    if parts[-2:] in (["white", "striped"], ["blue", "striped"], ["red", "striped"]):
        variants.add(_compact(f"{' '.join(parts[-2:])} {' '.join(parts[:-2])}"))
    return variants


_AUTHORITATIVE_VARIANTS = set()
for name in ALL_POKEMON:
    _AUTHORITATIVE_VARIANTS.update(_label_variants(name))


def _load_extra_catchable_2_names() -> List[str]:
    """Load names from hybrid mapping that are catchable == "2" in dataset_audit.json."""
    audit_path = _BASE_DIR / "reports" / "dataset_audit.json"
    catchable_2_norms = set()
    if audit_path.exists():
        try:
            with open(audit_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            official = data.get("official", {})
            dirs = official.get("directories", {})
            images = dirs.get("images", {}).get("manifest", [])
            shiny = dirs.get("shiny", {}).get("manifest", [])
            for r in images + shiny:
                if r.get("catchable") == "2":
                    catchable_2_norms.add(_compact(r.get("pokemon_name", "")))
                    catchable_2_norms.add(_compact(r.get("slug", "")))
        except Exception as e:
            logger.error("Failed to read dataset_audit.json: %s", e)

    hybrid_path = _BASE_DIR / "archive" / "experimental_hybrid" / "model" / "index_to_pokemon.json"
    extra_names = []
    if hybrid_path.exists():
        try:
            with open(hybrid_path, "r", encoding="utf-8") as f:
                hybrid_map = json.load(f)
            for name in hybrid_map.values():
                name_compact = _compact(name)
                if name_compact in catchable_2_norms:
                    if name_compact not in _AUTHORITATIVE_VARIANTS:
                        if name not in extra_names:
                            extra_names.append(name)
        except Exception as e:
            logger.error("Failed to read hybrid index_to_pokemon.json: %s", e)
            
    return sorted(extra_names)


EXTRA_POKEMON: List[str] = _load_extra_catchable_2_names()


def is_text_only_name(name: str) -> bool:
    """Return True if the name is a text-only catchable-2 Pokémon name."""
    comp = _compact(name)
    return comp not in _AUTHORITATIVE_VARIANTS



def resolve_authoritative_name(query: str) -> Optional[str]:
    """Return one canonical 936-label or extra catchable-2 name for an Assistant name.

    Matching accepts exact model labels and the safe human-form variants already
    used by the full-universe hint resolver.  It refuses to resolve when the
    authoritative mapping is unavailable, the name is unknown, or more than one
    canonical label matches.
    """
    if len(ALL_POKEMON) != _AUTHORITATIVE_LABEL_COUNT:
        logger.error(
            "Authoritative Pokémon mapping unavailable or incomplete: expected %d labels, found %d.",
            _AUTHORITATIVE_LABEL_COUNT,
            len(ALL_POKEMON),
        )
        return None

    normalized = _compact(query.strip())
    if not normalized:
        return None

    matches = [
        name for name in ALL_POKEMON
        if normalized in _label_variants(name)
    ]
    if len(matches) == 1:
        return matches[0]

    # Fallback to extra catchable=2 names for text-only matching
    extra_matches = [
        name for name in EXTRA_POKEMON
        if normalized in _label_variants(name)
    ]
    return extra_matches[0] if len(extra_matches) == 1 else None


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


# Helper functions (_compact, _label_variants, _FORM_PREFIXES) have been moved to the top of the file.


def _hint_pattern(pattern: str) -> tuple[str, int]:
    """Normalize a hint while preserving one-character wildcards."""
    pattern = pattern.replace("\\_", "_").replace("♀", "f").replace("♂", "m")
    normalized = unicodedata.normalize("NFKD", pattern)
    pieces: list[str] = []
    for char in normalized:
        if char in "_?":
            pieces.append(".")
        elif char.isalnum():
            pieces.append(re.escape(char.lower()))
    return "^" + "".join(pieces) + "$", len(pieces)


def _hint_matches(pattern: str, name: str) -> bool:
    """Match a normalized hint against all safe variants of one label."""
    regex, length = _hint_pattern(pattern)
    compiled = re.compile(regex)
    return any(len(variant) == length and compiled.fullmatch(variant) for variant in _label_variants(name))


def match_from_hint(pattern: str) -> List[str]:
    """Return every authoritative or text-only label matching *pattern*."""
    auth_matches = [name for name in ALL_POKEMON if _hint_matches(pattern, name)]
    if auth_matches:
        return auth_matches
    # Fallback to extra catchable=2 names for text-only matching
    return [name for name in EXTRA_POKEMON if _hint_matches(pattern, name)]


def get_best_hint_match(hint_text: str) -> Optional[str]:
    """Return exactly one full-universe match; abstain on ambiguity."""
    pattern = parse_hint(hint_text)
    if pattern is None:
        logger.debug("Could not parse hint from: %s", hint_text)
        return None

    matches = match_from_hint(pattern)
    logger.debug("Hint pattern '%s' matched %d authoritative labels: %s", pattern, len(matches), matches[:10])
    return matches[0] if len(matches) == 1 else None


def fuzzy_match(query: str, threshold: float = 0.6) -> Optional[str]:
    """Fuzzy-search *query* against ALL_POKEMON and EXTRA_POKEMON using SequenceMatcher.

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

    # Fallback to extra catchable=2 names for text-only matching
    best_extra_name: Optional[str] = None
    best_extra_ratio: float = 0.0
    for name in EXTRA_POKEMON:
        ratio = SequenceMatcher(None, query_lower, name.lower()).ratio()
        if ratio > best_extra_ratio:
            best_extra_ratio = ratio
            best_extra_name = name

    if best_extra_ratio > best_ratio and best_extra_ratio >= threshold:
        return best_extra_name

    return best_name if best_ratio >= threshold else None
