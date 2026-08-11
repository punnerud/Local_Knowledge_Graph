"""The graph as RDF, checked by a parser rather than by eye.

Every document these tests build is handed to rdflib, which fails on anything
malformed. That is the whole reason the dependency is here, and it earned itself
on the first run: the original Turtle emitted `lkg:run/abc123`, which is not a
legal prefixed name because a bare '/' cannot appear in one. It looked right. It
parsed as a syntax error, and the fix -- shorten only simple names, use @base for
the rest -- came from the parser rather than from a consumer discovering it later.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from mpe_lkg import rdf

rdflib = pytest.importorskip("rdflib", reason="the RDF validator is a dev dependency")

LKG = rdf.NS


def parse(text: str, fmt: str):
    graph = rdflib.Graph()
    graph.parse(data=text, format=fmt)
    return graph


def sample(**overrides):
    payload = {
        "graph": {"edges": [{"from": "Step1", "to": "Step2", "value": 0.8371}]},
        "answer": "13910400",
        "steps": [{"step": 1, "title": "Convert it", "content": "Ask for the whole thing."}],
        "conversions": ["23 week = 13910400 second"],
        "sums": ["23*604800 = 13910400"],
    }
    payload.update(overrides)
    return payload


def test_turtle_parses():
    assert len(parse(rdf.to_turtle("abc123", "How many seconds in 23 weeks?", **sample()),
                     "turtle")) > 20


def test_ntriples_parses():
    assert len(parse(rdf.to_ntriples("abc123", "q", **sample()), "nt")) > 20


def test_both_formats_carry_exactly_the_same_triples():
    """The streaming format and the stored one must not disagree about the facts."""
    turtle = parse(rdf.to_turtle("abc123", "q", **sample()), "turtle")
    ntriples = parse(rdf.to_ntriples("abc123", "q", **sample()), "nt")
    assert set(turtle) == set(ntriples)


def test_every_line_of_ntriples_parses_on_its_own():
    """The property that makes streaming possible, so it is tested rather than assumed.

    A consumer reading a partial stream must be able to parse what has arrived.
    That only holds if each line is a complete document, which is exactly what
    Turtle is not.
    """
    text = rdf.to_ntriples("abc123", "q", **sample())
    lines = [line for line in text.splitlines() if line.strip()]
    assert len(lines) > 20
    for line in lines:
        assert len(parse(line + "\n", "nt")) == 1, line


@pytest.mark.parametrize(
    "nasty",
    [
        'He said "hello"',
        "line one\nline two",
        "tab\there",
        "back\\slash",
        "carriage\r\nreturn",
        'everything: "\\\n\t at once',
    ],
)
def test_awkward_text_survives_the_round_trip(nasty: str):
    """A question is arbitrary text, and quotes and newlines are the ordinary case."""
    graph = parse(rdf.to_turtle("abc123", nasty), "turtle")
    questions = list(graph.objects(predicate=rdflib.URIRef(f"{LKG}question")))
    assert len(questions) == 1
    assert str(questions[0]) == nasty


def test_the_two_graphs_are_distinguishable_by_predicate():
    """The reason for lkg:basis: one graph is measured, the other is derived.

    A consumer that only trusts exact facts must be able to select them without
    heuristics, and a consumer that wants the associations must be able to keep
    them separate from the facts.
    """
    graph = parse(rdf.to_turtle("abc123", "q", **sample()), "turtle")
    basis = rdflib.URIRef(f"{LKG}basis")
    soft = set(graph.subjects(basis, rdflib.URIRef(f"{LKG}Embedding")))
    hard = set(graph.subjects(basis, rdflib.URIRef(f"{LKG}Exact")))
    assert len(soft) == 1, "the similarity link should be marked as an embedding"
    assert len(hard) == 2, "the conversion and the sum should both be marked exact"
    assert not (soft & hard), "nothing may be both measured and derived"


def test_a_conversion_becomes_two_unit_nodes_and_an_edge():
    """A sentence nothing can query is worth less than a graph something can."""
    graph = parse(rdf.to_turtle("abc123", "q", **sample()), "turtle")
    week = rdflib.URIRef(f"{LKG}unit/week")
    second = rdflib.URIRef(f"{LKG}unit/second")
    assert (week, rdflib.RDF.type, rdflib.URIRef(f"{LKG}Unit")) in graph
    assert (second, rdflib.RDF.type, rdflib.URIRef(f"{LKG}Unit")) in graph
    conversion = rdflib.URIRef(f"{LKG}run/abc123/conversion/1")
    assert (conversion, rdflib.URIRef(f"{LKG}from"), week) in graph
    assert (conversion, rdflib.URIRef(f"{LKG}to"), second) in graph
    assert str(graph.value(conversion, rdflib.URIRef(f"{LKG}toValue"))) == "13910400"


def test_a_non_terminating_fraction_is_not_dressed_up_as_a_decimal():
    """xsd:decimal has no notion of a third, and rounding one silently is a lie.

    The whole point of the exact layer is that a consumer can rely on it, so a
    value that cannot be written exactly is written as the ratio instead.
    """
    assert rdf.decimal(Fraction(1, 3)).startswith('"1/3"')
    assert "string" in rdf.decimal(Fraction(1, 3))
    # One that does terminate is a decimal, because it can be.
    assert rdf.decimal(Fraction(1, 4)) == '"0.25"^^<http://www.w3.org/2001/XMLSchema#decimal>'
    assert rdf.decimal(Fraction(20160)).startswith('"20160"')


def test_an_empty_run_is_still_a_valid_document():
    # A run killed before its first step must serialise, not raise.
    graph = parse(rdf.to_turtle("abc123", "q"), "turtle")
    assert (rdflib.URIRef(f"{LKG}run/abc123"), rdflib.RDF.type,
            rdflib.URIRef(f"{LKG}Run")) in graph


def test_a_title_with_a_space_does_not_produce_a_broken_iri():
    # Step titles are model output and end up near identifiers.
    text = rdf.to_ntriples(
        "abc 123", "q", graph={"edges": [{"from": "a b", "to": "c<d", "value": 0.5}]})
    assert len(parse(text, "nt")) > 0
