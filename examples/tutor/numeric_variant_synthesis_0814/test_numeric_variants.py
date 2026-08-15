#!/usr/bin/env python3
"""Dependency-light tests for the numeric-only reconstruction gate."""

from __future__ import annotations

import unittest

from numeric_variants import (
    apply_numeric_edits,
    mechanical_report,
    numeric_tokens,
    parse_tagged_json,
    strict_audit_pass,
)


class NumericVariantTests(unittest.TestCase):
    def test_exact_numeric_only_reconstruction(self) -> None:
        source = r"Find $x$ if $2x+3=11$."
        candidate, edits = apply_numeric_edits(
            source,
            [
                {"token_index": 0, "old": "2", "new": "4", "role": "coefficient"},
                {"token_index": 2, "old": "11", "new": "19", "role": "rhs"},
            ],
        )
        self.assertEqual(candidate, r"Find $x$ if $4x+3=19$.")
        self.assertTrue(all(mechanical_report(source, candidate, edits).values()))

    def test_duplicate_index_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            apply_numeric_edits(
                "Use 2 and 3.",
                [
                    {"token_index": 0, "old": "2", "new": "3"},
                    {"token_index": 0, "old": "2", "new": "4"},
                ],
            )

    def test_asymptote_numbers_are_protected(self) -> None:
        source = "Length is 4. [asy]draw((0,0)--(1,1));[/asy]"
        tokens = numeric_tokens(source)
        self.assertTrue(tokens[0]["editable"])
        self.assertTrue(all(not token["editable"] for token in tokens[1:]))
        with self.assertRaisesRegex(ValueError, "protected"):
            apply_numeric_edits(
                source, [{"token_index": 1, "old": "0", "new": "2"}]
            )

    def test_latex_exponents_and_subscripts_are_protected(self) -> None:
        tokens = numeric_tokens(r"Use $x^2+x_3+4$.")
        self.assertFalse(tokens[0]["editable"])
        self.assertFalse(tokens[1]["editable"])
        self.assertTrue(tokens[2]["editable"])

    def test_latex_thousands_suffix_is_protected(self) -> None:
        tokens = numeric_tokens(r"Target is $\$60,\!000$.")
        self.assertTrue(tokens[0]["editable"])
        self.assertFalse(tokens[1]["editable"])

    def test_large_scale_change_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "scale too much"):
            apply_numeric_edits(
                "Use 60 tickets.", [{"token_index": 0, "old": "60", "new": "5"}]
            )

    def test_invalid_latex_json_escape_is_repaired(self) -> None:
        raw = r'<FINAL_JSON>{"answer": "\sqrt{2}"}</FINAL_JSON>'
        self.assertEqual(parse_tagged_json(raw)["answer"], r"\sqrt{2}")

    def test_tagged_json_uses_last_block(self) -> None:
        raw = (
            '<FINAL_JSON>{"pass": false}</FINAL_JSON>\n'
            '<FINAL_JSON>{"pass": true}</FINAL_JSON>'
        )
        self.assertTrue(parse_tagged_json(raw)["pass"])

    def test_audit_requires_every_explicit_gate(self) -> None:
        keys = (
            "only_numeric_values_changed",
            "same_wording_units_target_constraints",
            "same_computation_graph",
            "same_knowledge_point",
            "structural_numbers_unchanged",
            "dependent_values_consistent",
            "well_posed",
            "similar_difficulty",
        )
        audit = {"pass": True, **{key: True for key in keys}}
        self.assertTrue(strict_audit_pass(audit))
        audit["well_posed"] = False
        self.assertFalse(strict_audit_pass(audit))


if __name__ == "__main__":
    unittest.main()
