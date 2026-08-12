"""A distributable battery of questions, generated rather than remembered.

The principle carried over from the original arithmetic battery, because it has
already caught real mistakes on both sides of the equals sign: the NUMBERS come
from a seeded RNG so no model has memorised them, and the TRUTH is computed by
the same code that renders the question, so the two cannot drift apart. A
hand-written expectation is a second chance to be wrong, and this project has
been bitten by exactly that once.

Domains are registered generators. Adding one is a pull request with two parts:
a module registering itself with ``@generator``, and tests proving its questions
self-grade. Three grading modes, in strictly decreasing order of authority:

* ``exact`` -- the answer is a Fraction computed at generation time. Graded by
  arithmetic; admits no argument.
* ``text`` -- the answer is one of a few accepted strings. Graded by substring;
  as good as the strings are.
* ``consistency`` -- no computable truth. Graded by whether the model agrees
  with ITSELF across repeated askings, which is evidence and not proof: a model
  can be confidently and consistently wrong, and this grading cannot see that.
  Results are reported apart from correctness, never mixed into it.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from fractions import Fraction

SEED = 20260811


@dataclass(frozen=True)
class Question:
    group: str
    text: str
    # The exact answer, as a Fraction so nothing is lost to float rounding.
    # None for text and consistency modes.
    answer: Fraction | None = None
    # The expression a reasoning step would hand over; built from the same
    # numbers as the text, so the evaluator can be checked against it.
    expression: str = ""
    mode: str = "exact"
    # Accepted substrings, for text mode.
    expect: tuple[str, ...] = ()
    domain: str = ""

    def matches(self, said: str) -> bool | None:
        """Is the answer present in what the model wrote?

        Generous about presentation, strict about the value: separators are
        stripped, but one digit out on a fifteen-digit product is a wrong
        answer, not a rounding difference. Returns None when this question
        cannot be graded this way at all (consistency mode), so a caller can
        never mistake "ungradeable" for "wrong".
        """
        if self.mode == "text":
            low = said.lower()
            return any(want.lower() in low for want in self.expect)
        if self.mode != "exact" or self.answer is None:
            return None
        cleaned = said.replace(",", "").replace(" ", "").replace("_", "")
        if self.answer.denominator == 1:
            return str(self.answer.numerator) in cleaned
        # 6 down to 2 decimal places. The top end matters as much as the
        # bottom: a model that answered -42.666667 for -128/3 was MORE precise
        # than the old 4-decimal ceiling, and was graded wrong for it. An
        # answer must never fail for exceeding the grader's precision.
        exact = float(self.answer)
        return any(f"{exact:.{p}f}".rstrip("0").rstrip(".") in cleaned
                   for p in (6, 5, 4, 3, 2))


Generator = Callable[[int, int], list[Question]]

GENERATORS: dict[str, Generator] = {}


def generator(name: str) -> Callable[[Generator], Generator]:
    """Register a domain. The name is what --domains selects by."""

    def register(fn: Generator) -> Generator:
        GENERATORS[name] = fn
        return fn

    return register


def domain_seed(seed: int, name: str) -> int:
    """A per-domain seed that is stable across machines and Python versions.

    NOT hash(): string hashing is randomised per process, and a battery that
    differs between two runs of the same seed measures nothing. crc32 is the
    same everywhere, forever.
    """
    return seed ^ zlib.crc32(name.encode())


def build(seed: int = SEED, per_group: int = 4,
          domains: Iterable[str] | None = None) -> list[Question]:
    """The battery: every registered domain, or the ones named.

    The arithmetic domain receives the RAW seed, deliberately: it predates the
    registry, and its questions for a given seed are pinned by recorded claims.
    Every other domain gets a seed derived from its name, so adding a domain
    never changes the questions of an existing one.
    """
    chosen = list(domains) if domains is not None else sorted(GENERATORS)
    out: list[Question] = []
    for name in chosen:
        if name not in GENERATORS:
            raise KeyError(f"no domain {name!r}; have {', '.join(sorted(GENERATORS))}")
        used = seed if name == "arithmetic" else domain_seed(seed, name)
        for question in GENERATORS[name](used, per_group):
            out.append(question if question.domain == name else
                       Question(**{**question.__dict__, "domain": name}))
    return out


def groups(domains: Iterable[str] | None = None) -> list[str]:
    return sorted({q.group for q in build(domains=domains)})


def truth_table(questions: list[Question]) -> str:
    """The battery as text, for pasting into a report."""
    width = max(len(q.group) for q in questions)
    lines = []
    for q in questions:
        right = f"{q.expression:>44} = {q.answer}" if q.mode == "exact" else f"[{q.mode}]"
        lines.append(f"{q.group:{width}}  {right}")
    return "\n".join(lines)


# Importing the domains registers them. At the bottom so the decorator exists.
from . import arithmetic, calculus, consistency, logic, physics, units  # noqa: E402, F401
