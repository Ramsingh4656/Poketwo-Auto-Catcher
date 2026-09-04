from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "bot"))
sys.path.insert(0, str(REPO_ROOT / "archive" / "experimental_hybrid" / "scripts"))

from pokemon_data import ALL_POKEMON, get_best_hint_match, match_from_hint, resolve_authoritative_name, is_text_only_name  # noqa: E402
from hybrid_inference import resolve_hint  # noqa: E402


class LiveHintResolverTests(unittest.TestCase):
    def test_authoritative_universe_is_the_deployed_onnx_universe(self) -> None:
        self.assertEqual(len(ALL_POKEMON), 936)
        self.assertEqual(len(set(ALL_POKEMON)), 936)

    def test_exact_hint(self) -> None:
        self.assertEqual(get_best_hint_match("The pokémon is **p i k a c h u**."), "pikachu")

    def test_partial_hint(self) -> None:
        self.assertEqual(get_best_hint_match("The pokémon is **p _ k a c h u**."), "pikachu")

    def test_spaces_punctuation_and_apostrophes_are_ignored(self) -> None:
        self.assertEqual(get_best_hint_match("The pokémon is **f a r f e t c h ' d**."), "farfetchd")
        self.assertEqual(get_best_hint_match("The pokémon is **m r - m i m e**."), "mr-mime")

    def test_regional_alias_matches_authoritative_label(self) -> None:
        self.assertEqual(
            get_best_hint_match("The pokémon is **g a l a r i a n a r t i c u n o**."),
            "articuno-galar",
        )

    def test_gender_symbol_matches_gendered_label(self) -> None:
        self.assertEqual(
            get_best_hint_match("The pokémon is **n i d o r a n ♀**."),
            "nidoran-f",
        )
        self.assertIn("nidoran-f", match_from_hint("n i d o r a n ♀"))

    def test_escaped_underscore_is_a_wildcard(self) -> None:
        self.assertEqual(get_best_hint_match(r"The pokémon is **p\_kachu**."), "pikachu")

    def test_ambiguous_hint_abstains_instead_of_picking_shortest(self) -> None:
        self.assertEqual(match_from_hint("n i d o r a n _"), ["nidoran-f", "nidoran-m"])
        self.assertIsNone(get_best_hint_match("The pokémon is **n i d o r a n _**."))

    def test_no_matching_class_abstains(self) -> None:
        self.assertIsNone(get_best_hint_match("The pokémon is **x y z q q q**."))

    def test_class_outside_authoritative_universe_does_not_match(self) -> None:
        # Gigantamax Alcremie exists in the experimental 1,659-label mapping,
        # but not in the deployed 936-label ONNX universe.
        outside = "Gigantamax Alcremie"
        self.assertNotIn(outside, ALL_POKEMON)
        compact = " ".join(outside)
        self.assertIsNone(get_best_hint_match(f"The pokémon is **{compact}**."))

    def test_extra_catchable_2_text_only_recognition(self) -> None:
        # Klang is not in the 936 visual universe (ALL_POKEMON)
        self.assertNotIn("Klang", ALL_POKEMON)
        self.assertNotIn("klang", ALL_POKEMON)

        # Klang is recognized by text/hint matches
        self.assertEqual(get_best_hint_match("The pokémon is **k l a n g**."), "Klang")
        self.assertEqual(resolve_authoritative_name("Klang"), "Klang")
        self.assertEqual(resolve_authoritative_name("klang"), "Klang")

        # It is correctly flagged as a text-only name
        self.assertTrue(is_text_only_name("Klang"))
        self.assertFalse(is_text_only_name("pikachu"))

        # Other catchable=2 names (like Aegislash) also resolve correctly
        self.assertEqual(get_best_hint_match("The pokémon is **a e g i s l a s h**."), "Aegislash")
        self.assertEqual(resolve_authoritative_name("Aegislash"), "Aegislash")


class HybridHintResolverTests(unittest.TestCase):
    def test_hybrid_resolver_uses_complete_supplied_universe(self) -> None:
        self.assertEqual(resolve_hint("p i k a c h u", ["eevee", "pikachu"]), "pikachu")

    def test_hybrid_resolver_abstains_on_ambiguity(self) -> None:
        self.assertIsNone(resolve_hint("n i d o r a n _", ["nidoran-f", "nidoran-m"]))


if __name__ == "__main__":
    unittest.main()
