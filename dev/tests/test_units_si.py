"""Composing an estimate across mixed units, which is where one falls apart.

The failure was diagnosed rather than guessed at, and it was not the refusal it
looked like. Given the same three figures, the assembly returned 4.08e10, 4.08e4
and 4.08e7 kilograms across three runs -- an identical mantissa every time. The
numbers were multiplied correctly and the EXPONENT was guessed from the units.
Sometimes the model noticed it could not do it and refused, which is what made it
look like a refusal problem.

Two fixes, each removing a guess rather than asking for care:

* restate every quantity in SI before composing -- spread went from 6 orders of
  magnitude to 0
* derive the unit the answer must carry, since m^2 * m * kg/m^3 is a calculation
  and not an opinion -- the answer stopped saying "grams" where it meant
  kilograms, a thousandfold error with the arithmetic untouched

Together: 4.08e16 kg, three times identically, against an accepted 1.3e16.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from mpe_lkg.arithmetic import in_si, product_unit, restate


class TestRestating:
    @pytest.mark.parametrize(
        ("written", "expected", "si"),
        [
            ("510,072,000 square kilometers", Fraction(510_072_000) * 10**6, "m^2"),
            ("100 kilometers", Fraction(100_000), "m"),
            ("0.8 grams per cubic metre", Fraction(8, 10000), "kg/m^3"),
            ("5.15 kilograms", Fraction(515, 100), "kg"),
        ],
    )
    def test_quantities_come_back_in_base_units(self, written, expected, si):
        found = in_si(written)
        assert found, f"nothing found in {written!r}"
        _, value, unit = found[0]
        assert value == expected
        assert unit == si

    def test_a_unit_it_does_not_know_is_left_alone(self):
        """An incomplete table that refuses beats a complete-looking one that guesses."""
        assert in_si("3 furlongs") == []
        assert restate("3 furlongs") == "3 furlongs"

    def test_text_without_quantities_is_untouched(self):
        assert restate("The capital of France is Paris.") == "The capital of France is Paris."

    def test_the_original_wording_is_kept_beside_the_si_form(self):
        out = restate("The area is 510,072,000 square kilometers.")
        assert "510,072,000 square kilometers" in out
        assert "m^2" in out


class TestTheOutputUnit:
    def test_the_unit_of_the_product_is_calculated(self):
        """m^2 * m * kg/m^3 = kg. Not an opinion."""
        assert product_unit(
            "510072000 square kilometers, 100 kilometers, 0.8 grams per cubic metre") == "kg"

    def test_two_lengths_make_an_area(self):
        assert product_unit("5 metres and 3 metres") == "m^2"

    def test_one_quantity_has_no_product(self):
        assert product_unit("100 kilometers") == ""

    def test_nothing_measurable_has_no_unit(self):
        assert product_unit("Paris is the capital of France") == ""

    def test_exponents_add_and_cancel(self):
        # m * kg/m^3 is kg/m^2, not "metres times a density". My first expectation
        # here was wrong and the code was right, which is the useful direction.
        assert product_unit("6 metres and 0.5 kilograms per cubic metre") == "kg/m^2"
        # And a length times an area is a volume.
        assert product_unit("6 metres and 2 square metres") == "m^3"
