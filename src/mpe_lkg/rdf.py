"""The reasoning graph as RDF, in Turtle and N-Triples.

The picture is the point of this project, but a picture is where the data stops.
RDF is where it carries on: a run becomes triples another tool can query, join to
something else, or keep. Nothing here depends on a library -- the serialisation is
written out directly, and ``rdflib`` appears only in the tests, where it parses
what this produces and fails on anything malformed.

**Two graphs, and they are not the same kind of thing.** That distinction is the
reason for the ``lkg:basis`` predicate rather than a comment:

``lkg:Embedding``
    Steps joined by how similar their embeddings are. Soft: it is a measurement of
    an association, it has no truth value, and two steps at 0.83 are not related
    by any fact. It is what makes indirect knowledge visible -- the reason a step
    about coastal winters sits near one about ocean currents.

``lkg:Exact``
    Facts settled by mpeqs: an arithmetic result, or a conversion between two
    units. Hard: derived from a graph of exact ratios, reproducible, and wrong
    only if the ratio table is wrong. A consumer can rely on these in a way it
    can never rely on a cosine.

Keeping both in one document, distinguishable by predicate, is what lets a reader
see the reasoning and a machine take only the part it can trust.

Two formats, for two purposes:

* **N-Triples** -- one complete triple per line, no prefixes, no state. A line is
  valid on its own, so a run can be streamed as it happens and a consumer can
  parse what has arrived so far. This is the format for the live endpoint.
* **Turtle** -- prefixed and grouped, for reading and for keeping. Needs the whole
  document, so it is what a finished run serialises to.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from fractions import Fraction

# Ends in '/', not '#', and that is load-bearing. With a '#' namespace, @base
# resolution follows RFC 3986 and a relative <run/abc123> replaces the last path
# segment -- so the "ns" disappeared and the Turtle described different subjects
# than the N-Triples did. The test comparing the two formats caught it; by eye
# both looked fine.
NS = "https://github.com/punnerud/Local_Knowledge_Graph/ns/"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
XSD = "http://www.w3.org/2001/XMLSchema#"

PREFIXES = {
    "lkg": NS,
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "xsd": XSD,
}

# Turtle and N-Triples both take these six escapes in a quoted literal. Order
# matters: the backslash has to go first or it re-escapes the escapes.
_ESCAPES = [("\\", "\\\\"), ('"', '\\"'), ("\n", "\\n"),
            ("\r", "\\r"), ("\t", "\\t"), ("\b", "\\b")]

# An IRI may not contain these. A step title is user-adjacent text and ends up in
# a run identifier, so this is not hypothetical.
_UNSAFE_IRI = re.compile(r'[\x00-\x20<>"{}|^`\\]')


def literal(text: str) -> str:
    """A quoted literal, escaped so a parser accepts it."""
    out = str(text)
    for raw, escaped in _ESCAPES:
        out = out.replace(raw, escaped)
    return f'"{out}"'


def decimal(value) -> str:
    """A typed decimal. Fractions become their exact decimal where one exists.

    ``xsd:decimal`` has no notion of a third, so a non-terminating fraction is
    written as the ratio it is, typed as a string, rather than silently rounded
    into a number a consumer would take as exact.
    """
    if isinstance(value, Fraction):
        if value.denominator == 1:
            return f'"{value.numerator}"^^<{XSD}integer>'
        # A fraction terminates in decimal only if its denominator is 2^a * 5^b.
        reduced = value.denominator
        for prime in (2, 5):
            while reduced % prime == 0:
                reduced //= prime
        if reduced != 1:
            return f'"{value.numerator}/{value.denominator}"^^<{XSD}string>'
        value = float(value)
    text = f"{float(value):.12g}"
    return f'"{text}"^^<{XSD}decimal>'


def iri(*parts: str) -> str:
    """An IRI under this run's namespace, with anything unsafe percent-escaped."""
    tail = "/".join(_UNSAFE_IRI.sub(lambda m: f"%{ord(m.group()):02X}", str(p)) for p in parts)
    return f"<{NS}{tail}>"


def triple(subject: str, predicate: str, obj: str) -> str:
    """One N-Triples line, terminator included."""
    return f"{subject} {predicate} {obj} ."


def _p(name: str) -> str:
    return f"<{NS}{name}>"


def run_triples(
    run_id: str,
    question: str,
    *,
    graph: dict | None = None,
    answer: str = "",
    steps: list[dict] | None = None,
    conversions: list[str] | None = None,
    sums: list[str] | None = None,
    session: str = "",
    findings: list[dict] | None = None,
    votes: list[dict] | None = None,
    agreement: dict | None = None,
) -> Iterator[str]:
    """Every triple for a run, in an order that streams sensibly.

    The run first, then each step as it would have arrived, then the links. A
    consumer reading the stream sees a step before anything that refers to it.
    """
    run = iri("run", run_id)
    yield triple(run, f"<{RDF_TYPE}>", _p("Run"))
    yield triple(run, _p("question"), literal(question))
    if answer:
        yield triple(run, _p("answer"), literal(answer))
    if session:
        yield triple(run, _p("session"), literal(session))

    for step in steps or []:
        number = step.get("step") or step.get("number") or 0
        node = iri("run", run_id, "step", str(number))
        yield triple(run, _p("step"), node)
        yield triple(node, f"<{RDF_TYPE}>", _p("Step"))
        yield triple(node, _p("number"), f'"{int(number)}"^^<{XSD}integer>')
        if step.get("title"):
            yield triple(node, f"<{RDFS_LABEL}>", literal(step["title"]))
        if step.get("content"):
            yield triple(node, _p("content"), literal(step["content"]))

    for edge in (graph or {}).get("edges", []):
        source, target = str(edge.get("from", "")), str(edge.get("to", ""))
        if not source or not target:
            continue
        link = iri("run", run_id, "link", f"{source}-{target}")
        yield triple(link, f"<{RDF_TYPE}>", _p("Link"))
        yield triple(link, _p("from"), iri("run", run_id, "node", source))
        yield triple(link, _p("to"), iri("run", run_id, "node", target))
        yield triple(link, _p("similarity"), decimal(edge.get("value", 0.0)))
        # The soft graph: an association, with no truth value.
        yield triple(link, _p("basis"), _p("Embedding"))
        yield triple(run, _p("link"), link)

    # The hard graph. Every one of these is settled by an exact evaluator, and a
    # consumer can rely on it in a way it can never rely on a cosine.
    for index, statement in enumerate(conversions or [], 1):
        node = iri("run", run_id, "conversion", str(index))
        yield triple(node, f"<{RDF_TYPE}>", _p("Conversion"))
        yield triple(node, _p("statement"), literal(statement))
        yield triple(node, _p("basis"), _p("Exact"))
        yield triple(run, _p("fact"), node)
        for subject, predicate, obj in _conversion_parts(run_id, index, statement):
            yield triple(subject, predicate, obj)

    for index, statement in enumerate(sums or [], 1):
        node = iri("run", run_id, "sum", str(index))
        yield triple(node, f"<{RDF_TYPE}>", _p("Calculation"))
        yield triple(node, _p("statement"), literal(statement))
        yield triple(node, _p("basis"), _p("Exact"))
        yield triple(run, _p("fact"), node)

    # The structure of an exploration: which questions the question became, and
    # what each one came back with. Verified missing before being added -- a
    # settle run produced branches, findings, a vote and an agreement, and the
    # serialisation dropped every one, so a consumer could not tell an agreed
    # answer from a single run's synthesis.
    for index, finding in enumerate(findings or [], 1):
        node = iri("run", run_id, "finding", str(index))
        yield triple(node, f"<{RDF_TYPE}>", _p("Finding"))
        yield triple(node, _p("question"), literal(finding.get("question", "")))
        yield triple(node, _p("answer"), literal(finding.get("answer", "")))
        yield triple(run, _p("finding"), node)

    # The vote is opinion and is marked as such: three readings of two answers,
    # not a fact about the world. lkg:basis distinguishes it the same way the
    # colours do.
    for index, tally in enumerate(votes or [], 1):
        node = iri("run", run_id, "vote", str(index))
        yield triple(node, f"<{RDF_TYPE}>", _p("Vote"))
        yield triple(node, _p("agreeing"), f'"{int(tally.get("agree", 0))}"^^<{XSD}integer>')
        yield triple(node, _p("dissenting"), f'"{int(tally.get("disagree", 0))}"^^<{XSD}integer>')
        yield triple(node, _p("basis"), _p("Opinion"))
        if tally.get("about"):
            yield triple(node, _p("about"), literal(tally["about"]))
        yield triple(run, _p("vote"), node)

    if agreement:
        node = iri("run", run_id, "agreement")
        yield triple(node, f"<{RDF_TYPE}>", _p("Agreement"))
        # "wording" when two runs matched outright, "vote" when a majority said
        # so. A consumer weighing the answer is entitled to the difference.
        yield triple(node, _p("by"), literal(agreement.get("by", "vote")))
        if agreement.get("round"):
            yield triple(node, _p("round"),
                         f'"{int(agreement["round"])}"^^<{XSD}integer>')
        yield triple(run, _p("agreement"), node)


# "23 week = 13910400 second" -- taken apart so the units become nodes and the
# conversion becomes an edge between them, rather than a sentence nothing can query.
_STATEMENT = re.compile(
    r"^\s*(?P<from_value>\S+)\s+(?P<from_unit>\w+)\s*=\s*"
    r"(?P<to_value>\S+)\s+(?P<to_unit>\w+)\s*$")


def _conversion_parts(run_id: str, index: int, statement: str):
    match = _STATEMENT.match(statement)
    if not match:
        return
    node = iri("run", run_id, "conversion", str(index))
    source = iri("unit", match.group("from_unit"))
    target = iri("unit", match.group("to_unit"))
    yield source, f"<{RDF_TYPE}>", _p("Unit")
    yield source, f"<{RDFS_LABEL}>", literal(match.group("from_unit"))
    yield target, f"<{RDF_TYPE}>", _p("Unit")
    yield target, f"<{RDFS_LABEL}>", literal(match.group("to_unit"))
    yield node, _p("from"), source
    yield node, _p("to"), target
    try:
        yield node, _p("fromValue"), decimal(Fraction(match.group("from_value")))
        yield node, _p("toValue"), decimal(Fraction(match.group("to_value")))
    except (ValueError, ZeroDivisionError):
        # A value that will not parse is left off rather than guessed at. The
        # statement literal is still there and still readable.
        return


def to_ntriples(*args, **kwargs) -> str:
    """The whole run as N-Triples: one triple per line, no prefixes."""
    return "\n".join(run_triples(*args, **kwargs)) + "\n"


def to_turtle(*args, **kwargs) -> str:
    """The whole run as Turtle.

    Prefixed for reading. The triples themselves are written out one per line
    rather than grouped with ';' -- Turtle allows both, and a diff between two
    runs is worth more than a few saved bytes.
    """
    # @base, so a run IRI reads as <run/abc123> rather than sixty characters of
    # namespace repeated on every line. The vocabulary still uses prefixes: those
    # are simple names and a prefixed name is the shorter form for them.
    lines = [f"@base <{NS}> ."]
    lines += [f"@prefix {name}: <{url}> ." for name, url in sorted(PREFIXES.items())]
    lines.append("")
    for statement in run_triples(*args, **kwargs):
        lines.append(_shorten(statement))
    return "\n".join(lines) + "\n"


# A prefixed name may not contain an unescaped '/', so only IRIs whose local part
# is a simple name can be shortened. rdflib rejected the first version of this
# over exactly that -- `lkg:run/abc123` is not Turtle, and the validator said so
# before a single consumer had to find out.
_SIMPLE_LOCAL = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")


def _shorten(statement: str) -> str:
    def replace(match):
        full = match.group(1)
        for name, url in PREFIXES.items():
            if full.startswith(url):
                local = full[len(url):]
                if _SIMPLE_LOCAL.match(local):
                    return f"{name}:{local}"
                # Not a legal prefixed name, but it IS legal relative to @base.
                return f"<{local}>"
        return f"<{full}>"

    return re.sub(r"<([^>]+)>", replace, statement)
