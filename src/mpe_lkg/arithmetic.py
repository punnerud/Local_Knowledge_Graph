"""Check the sums a model writes down, without asking it to check them.

Measured on this eval: pushing runs to eight steps lifted hard questions from
50 % to 62 % but dropped straightforward multi-step arithmetic from 94 % to 67 %.
The failures were not reasoning failures. They were "10080 minutes in a
fortnight" (that is a week) and "6.00 change" from a 20 note on 13.50 of goods.

So the arithmetic is handed to something that does arithmetic. ``mpeqs`` evaluates
exactly, in fractions, and refuses rather than guessing. This module finds the
sums in a step's prose and asks it.

Two rules keep this from doing harm, because a wrong correction is worse than no
correction:

* **Only unambiguous claims are checked.** The left side must be arithmetic and
  nothing else -- digits, operators, brackets. ``x = 5`` is a definition, not a
  claim about a sum, and is left alone.
* **Rounding is not an error.** A model writing 28.27 for 28.2743 has rounded
  correctly. The tolerance comes from how many decimals it chose to write, so
  three decimals are held to three and none to a half.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

# A claim is an arithmetic expression, then = or "is", then a number. The left
# side must contain an operator: without one there is no sum to check.
CLAIM = re.compile(
    # An expression may open with a bracket -- "(17/100) * 250" is the shape a
    # model writes a percentage in, and it was the first thing this missed.
    r"(?P<expr>\(?[\d.]+\)?(?:\s*[+\-*/^]\s*\(?[\d.()]+\)?)+)"
    r"\s*(?:=|equals?|is)\s*"
    r"(?P<stated>-?[\d.]+)",
    re.IGNORECASE,
)

# What models write instead of what Python parses.
SUBSTITUTIONS = {"×": "*", "·": "*", "÷": "/", "−": "-", "–": "-", "^": "**"}


@dataclass
class Claim:
    expression: str
    stated: str
    exact: Fraction
    ok: bool

    def describe(self) -> str:
        return f"{self.expression} = {float(self.exact):g}, not {self.stated}"


def _normalise(text: str) -> str:
    for wrong, right in SUBSTITUTIONS.items():
        text = text.replace(wrong, right)
    # Thousands separators, but only between digits: "20,160" is a number while
    # "3, 4 and 5" is a list.
    return re.sub(r"(?<=\d),(?=\d\d\d\b)", "", text)


def _tolerance(stated: str) -> Fraction:
    """Half of the last place the model chose to write.

    Someone who writes 28.27 has claimed a value to two decimals and is right if
    the exact answer rounds there. Someone who writes 28.2743 has claimed more and
    is held to it.
    """
    if "." not in stated:
        return Fraction(1, 2)
    return Fraction(1, 2) / (10 ** len(stated.split(".")[1]))


def check(text: str, *, limit: int = 6) -> list[Claim]:
    """Every arithmetic claim in the text, with whether it holds."""
    try:
        import mpeqs
    except ImportError:
        # A declared dependency, so this should not happen. Kept because a broken
        # install showing a working app minus the exact sums beats a stack trace.
        return []

    found: list[Claim] = []
    for match in CLAIM.finditer(_normalise(text)):
        expression = match.group("expr").strip()
        stated = match.group("stated").strip().rstrip(".")
        if not stated or len(found) >= limit:
            break
        try:
            exact = Fraction(mpeqs.solve({"solver": "arith", "answer": expression}))
            claimed = Fraction(stated)
        except (mpeqs.Refusal, ValueError, ZeroDivisionError, ArithmeticError):
            # A refusal means the record could not derive it, which is not the
            # same as the model being wrong. Say nothing.
            continue
        found.append(Claim(expression, stated, exact, abs(exact - claimed) <= _tolerance(stated)))
    return found


def errors(text: str) -> list[Claim]:
    return [claim for claim in check(text) if not claim.ok]


def correction(bad: list[Claim]) -> str:
    """What to tell the model, naming the sum rather than the conclusion."""
    lines = "\n".join(f"- {claim.describe()}" for claim in bad)
    return (
        f"One of those sums is wrong:\n{lines}\n\n"
        "Redo this step with the corrected figure. Change only what the arithmetic "
        "changes; keep the rest of your reasoning."
    )


def evaluate(expression: str):
    """Evaluate an expression the model handed over, or None if it will not.

    This is the other half of the same idea as ``check``, and the stronger half.
    Rather than searching prose for something shaped like a sum, the step schema
    asks for the expression outright, so what arrives is already machine-readable.
    """
    text = _normalise(str(expression or "")).strip().rstrip("=").strip()
    if not text or not any(op in text for op in "+-*/"):
        return None
    try:
        import mpeqs

        return Fraction(mpeqs.solve({"solver": "arith", "answer": text}))
    except ImportError:
        return None  # See check(): a declared dependency, defended against anyway.
    except Exception:
        # A refusal means the record could not derive it, which is not evidence
        # about the model. Say nothing rather than guessing.
        return None


def as_text(value: Fraction) -> str:
    """A value a reader recognises: 42.5 rather than 85/2, 20160 rather than 20160/1."""
    if value.denominator == 1:
        return str(value.numerator)
    # %g alone was wrong here: it defaults to six significant digits, so the exact
    # 1609.344 metres in a mile printed as 1609.34 and quietly threw away a digit
    # the whole point of this module was to keep. Twelve significant digits, then
    # trailing zeros stripped.
    text = f"{round(float(value), 6):.12g}"
    return text.rstrip("0").rstrip(".") if "." in text else text



# "23 weeks to seconds", "1 mile in metres", "convert 5 kg into grams".
CONVERSION = re.compile(
    # The value is OPTIONAL, and that was not a guess: asked for a conversion the
    # model wrote "weeks to seconds", with no number at all, and a regex demanding
    # one matched nothing. Bare units are answered with the factor for one -- which
    # is the thing it was getting wrong anyway, having built it by hand as (7*24)*23.
    r"(?:(?P<value>-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?)\s*)?"
    r"(?P<source>[A-Za-z_]+)\s*"
    # "1 kg OF FEATHERS to grams" -- the substance is read past and dropped, since
    # it is not a dimension and cannot affect the factor.
    r"(?:\s+of\s+[A-Za-z_ ]+?)?\s*"
    r"(?:->|→|to|in|into|as)\s+"
    r"(?P<target>[A-Za-z_]+)",
    re.IGNORECASE,
)


def convert(request: str):
    """Do a named unit conversion exactly, or return None.

    The counterpart to ``evaluate``, and needed for the same reason one level up:
    an exact evaluator settles what an expression comes to, and cannot tell
    whether the expression meant anything. Measured on unit questions, the model
    scored 0 of 4 WITH exact arithmetic available -- it wrote 604800/161, which is
    seconds-in-a-week divided by a count of days, and mpeqs returned exactly the
    wrong number it asked for.

    So the factor is not the model's to remember. It comes from a graph of exact
    ratios, where a conversion is a path and its value is the product along it.

    Returns ``(text, value, label)`` -- the readable statement, the exact Fraction,
    and what the value IS in the question's own words ("second in 23 week"). The
    label is what makes the value selectable at the end: an unlabelled number
    cannot be matched to a question, which was measured at 0 of 5.

    None if there is nothing here to do or the units are unknown.
    """
    match = CONVERSION.search(_normalise(str(request or "")))
    if not match:
        return None
    try:
        from mpeqs import units
    except ImportError:
        return None
    raw = (match.group("value") or "1").replace(" ", "")
    try:
        value = units.convert(Fraction(raw), match.group("source"), match.group("target"))
        source = units.canonical(match.group("source"))
        target = units.canonical(match.group("target"))
    except Exception:
        # An unknown unit or a cross-dimension request is a refusal, not a guess.
        # Saying nothing is the safe failure, exactly as it is for a wrong sum.
        return None
    return (f"{raw} {source} = {readable(value)} {target}", value,
            f"{target} in {raw} {source}")


def readable(value: Fraction) -> str:
    """The same value, grouped in thousands, for anything a model has to read back.

    Not cosmetic. Llama-3 chunks a digit run into groups of three LEFT to right,
    so 13910400 tokenises as [139][104][00] and comes back out as "139,104,000" --
    the model is faithfully copying a value it has mis-segmented. Writing the
    separators ourselves forces the grouping to align with the value.

    Reproduced on this machine: 77.5% correct with bare digits, 96.2% with
    separators, across eight magnitudes. as_text stays unseparated, because that
    is what gets compared and stored.
    """
    if value.denominator == 1:
        return f"{value.numerator:,}"
    text = as_text(value)
    whole, _, rest = text.partition(".")
    try:
        grouped = f"{int(whole):,}"
    except ValueError:
        return text
    return f"{grouped}.{rest}" if rest else grouped


# A quantity written in prose: "510,072,000 square kilometers", "0.8 grams per
# cubic metre". Compound units are matched whole, because it is exactly the
# compound ones the model gets wrong.
QUANTITY_IN_TEXT = re.compile(
    r"(?P<value>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<unit>(?:square|cubic)\s+\w+|\w+(?:\s+per\s+(?:square|cubic)\s+\w+)?)",
    re.IGNORECASE,
)

# What each written unit is in SI base units, as (factor, si name). Only the ones
# a physical estimate actually reaches for -- an incomplete table that refuses
# what it does not know beats a complete-looking one that guesses.
SI = {
    "kilometer": (1000, "m"), "kilometre": (1000, "m"), "km": (1000, "m"),
    "meter": (1, "m"), "metre": (1, "m"), "m": (1, "m"),
    "centimeter": (Fraction(1, 100), "m"), "centimetre": (Fraction(1, 100), "m"),
    "square kilometer": (10**6, "m^2"), "square kilometre": (10**6, "m^2"),
    "square meter": (1, "m^2"), "square metre": (1, "m^2"),
    "cubic kilometer": (10**9, "m^3"), "cubic kilometre": (10**9, "m^3"),
    "cubic meter": (1, "m^3"), "cubic metre": (1, "m^3"),
    "kilogram": (1, "kg"), "kg": (1, "kg"),
    "gram": (Fraction(1, 1000), "kg"), "grams": (Fraction(1, 1000), "kg"),
    "tonne": (1000, "kg"), "tonnes": (1000, "kg"),
    "gram per cubic metre": (Fraction(1, 1000), "kg/m^3"),
    "gram per cubic meter": (Fraction(1, 1000), "kg/m^3"),
    "grams per cubic metre": (Fraction(1, 1000), "kg/m^3"),
    "grams per cubic meter": (Fraction(1, 1000), "kg/m^3"),
    "kilogram per cubic metre": (1, "kg/m^3"),
    "kilograms per cubic metre": (1, "kg/m^3"),
}


def in_si(text: str) -> list[tuple[str, Fraction, str]]:
    """Every quantity in the text, restated in SI base units.

    Composition across mixed units is where an estimate falls apart, and it falls
    apart quietly: measured, the same computation came back as 4.08e10, 4.08e16
    and 4.08e7 kilograms across three runs. The mantissa was identical every time
    -- the model multiplied the numbers correctly and guessed the exponent from
    the units. Restating everything in metres and kilograms first removes the
    guess rather than asking it to be careful.
    """
    out = []
    for match in QUANTITY_IN_TEXT.finditer(_normalise(text)):
        unit = " ".join(match.group("unit").lower().split())
        entry = SI.get(unit) or SI.get(unit.rstrip("s"))
        if entry is None:
            continue
        factor, si = entry
        try:
            value = Fraction(match.group("value").replace(",", ""))
        except (ValueError, ZeroDivisionError):
            continue
        out.append((match.group(0), value * Fraction(factor), si))
    return out


def restate(text: str) -> str:
    """The text with each quantity followed by its SI form, once."""
    seen = set()
    additions = []
    for written, value, si in in_si(text):
        key = (str(value), si)
        if key in seen:
            continue
        seen.add(key)
        additions.append(f"{written} = {readable(value)} {si}")
    if not additions:
        return text
    return text + "  [in SI: " + "; ".join(additions) + "]"


def _dimensions(si: str) -> dict[str, int]:
    """A unit like 'kg/m^3' as exponents: {'kg': 1, 'm': -3}."""
    top, _, bottom = si.partition("/")
    out: dict[str, int] = {}
    for part, sign in ((top, 1), (bottom, -1)):
        for token in part.split("*"):
            token = token.strip()
            if not token:
                continue
            base, _, power = token.partition("^")
            out[base] = out.get(base, 0) + sign * int(power or 1)
    return {k: v for k, v in out.items() if v}


def _render(dimensions: dict[str, int]) -> str:
    top = [b if e == 1 else f"{b}^{e}" for b, e in sorted(dimensions.items()) if e > 0]
    bottom = [b if e == -1 else f"{b}^{-e}" for b, e in sorted(dimensions.items()) if e < 0]
    if not top and not bottom:
        return "dimensionless"
    return "/".join(["*".join(top) or "1"] + (["*".join(bottom)] if bottom else []))


def product_unit(text: str) -> str:
    """The unit you get by multiplying every quantity in the text together.

    Measured, and it is the last guess left in a Fermi estimate: given the same
    figures in SI the model composed 4.08e16 correctly every time and then wrote
    "grams" where the answer is kilograms -- a thousandfold error in the label,
    with the arithmetic untouched. m^2 * m * kg/m^3 is kg, and that is a
    calculation rather than an opinion, so it is done here.

    Empty when there is nothing to multiply, which is most questions.
    """
    units = [si for _, _, si in in_si(text)]
    if len(units) < 2:
        return ""
    total: dict[str, int] = {}
    for si in units:
        for base, power in _dimensions(si).items():
            total[base] = total.get(base, 0) + power
    return _render({k: v for k, v in total.items() if v})


# A question that IS one conversion: "How many seconds are there in 82 weeks?"
QUESTION_CONVERSION = re.compile(
    r"how\s+(?:many|much)\s+(?P<target>[A-Za-z_]+)\s+(?:is|are)\s+(?:there\s+)?in\s+"
    r"(?P<value>-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?)\s*(?P<source>[A-Za-z_]+)\s*\??",
    re.IGNORECASE,
)


def question_conversion(question: str):
    """If the whole question is a single unit conversion, settle it up front.

    Measured, and the reason this exists: on twelve seconds-in-N-weeks
    questions the model asserted a bare number eight times -- "There are
    1,612,800 seconds", no expression, no calc, no convert -- and a bare
    assertion gives every gate in the pipeline nothing to hold on to. When the
    question itself parses as one conversion, the exact value can exist BEFORE
    the first model call, and the run starts anchored instead of hoping the
    model asks.

    Returns the same (text, value, label) as ``convert``, or None when the
    question is not a plain conversion -- which is almost every question, and
    the reason this stays a narrow shape rather than a router.
    """
    match = QUESTION_CONVERSION.search(_normalise(str(question or "")))
    if not match:
        return None
    return convert(f"{match.group('value')} {match.group('source')} "
                   f"to {match.group('target')}")


# "d/dx (3*x**2+5)**4 at x=1/2", "derivative of x**3 - 4*x at 2".
DERIVATIVE = re.compile(
    r"(?:d/dx|derivative\s+of)\s*(?P<expr>.+?)\s+at\s*(?:x\s*=\s*)?"
    r"(?P<at>-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?)\s*$",
    re.IGNORECASE,
)


def derivative_request(request: str):
    """A derivative the model asked for, settled exactly, or None.

    The same contract as ``convert`` one shelf over: the model NAMES the
    derivative and the value comes from mpeqs.calculus, which computes it two
    independent ways and whose tests demand they agree. Measured need: on the
    calculus battery the model dropped a chain factor (6912 where the answer
    is 235824) and read a tangent at the wrong point -- the same
    guessed-instead-of-asked failure the calc field closed for arithmetic.

    Returns ``(text, value, label)`` or None; a refusal from the calculus
    module -- x**x, an unknown name -- is silence rather than a guess.
    """
    match = DERIVATIVE.search(_normalise(str(request or "")))
    if not match:
        return None
    try:
        from mpeqs import calculus
    except ImportError:
        return None
    try:
        at = Fraction(match.group("at").replace(" ", ""))
        expression = match.group("expr").strip()
        value = calculus.derivative_at(expression, at=at)
    except Exception:
        return None
    return (f"d/dx {expression} at x={at} = {readable(value)}", value,
            f"derivative of {expression} at x={at}")
