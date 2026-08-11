# Three ways to answer, and the graph as RDF

## Modes

```bash
curl -sX POST localhost:5100/jobs -H 'content-type: application/json' \
     -d '{"query":"...","mode":"settle"}'
```

| mode | what it does |
|---|---|
| `reason` | one run — steps, a graph, an answer |
| `explore` | the question is split into questions, **each answered by a run of its own**, then assembled |
| `settle` | explored, explored **again**, and finished only when two independent runs agree |

The same picker sits on the web page.

`settle` exists so that no single call decides. When the two runs disagree, three checks vote
through different lenses — same value, same conclusion, same action — and the tally is shown
rather than reduced to a verdict, because 2-1 and 3-0 are different things.

A vote is still opinion, so both answers are then **probed**. The same sum with its numbers
moved is graded against the exact evaluator and settles the matter outright; asking what a
knower would also know is suggestive only, and labelled as such, because a model can be
confidently and consistently wrong.

`explore` splits with tapering breadth — 5 questions at the top, then 4, 3, 2 — so a deep
exploration is large but still ends. Two embedding guards keep it honest, and neither is the
model's to apply: anything that has **drifted** from the parent question is dropped (the model
cannot tell — each step looks reasonable from the one before), and anything **already asked**
is dropped (it cannot tell — it does not hold the earlier questions).

## Headless jobs

```bash
ID=$(curl -sX POST localhost:5100/jobs -H 'content-type: application/json' \
       -d '{"query":"How many seconds are there in 23 weeks?"}' | jq -r .id)

curl -s localhost:5100/jobs/$ID              # {"state":"running","steps":3,...}
curl -s localhost:5100/jobs/$ID/stream       # N-Triples, live, one triple per line
curl -s localhost:5100/jobs/$ID/rdf          # Turtle, once it has finished
curl -sX DELETE localhost:5100/jobs/$ID      # stop it
```

## The RDF

Two formats for two purposes. **N-Triples** streams: each line is a complete document, so a
consumer can parse what has arrived without waiting for the end. **Turtle** is prefixed and
readable, and needs the whole document, so it is what a finished run serialises to.

The `lkg:basis` predicate separates kinds of claim, so a consumer can take only the part it
can rely on:

| basis | meaning |
|---|---|
| `lkg:Embedding` | measured similarity — an association, no truth value |
| `lkg:Exact` | settled by the exact evaluator — reproducible |
| `lkg:Opinion` | a vote — three model readings, counted |

```turtle
<run/a1b2/link/Step1-Step2>  lkg:similarity "0.8371"^^xsd:decimal ;
                             lkg:basis      lkg:Embedding .

<run/a1b2/conversion/1>      lkg:statement  "23 week = 13910400 second" ;
                             lkg:from       <unit/week> ;
                             lkg:to         <unit/second> ;
                             lkg:basis      lkg:Exact .
```

An exploration also carries its structure: `lkg:Finding` (each sub-question and what it came
back with), `lkg:Vote` (the tally and what the dispute was about), and `lkg:Agreement` (how
the run finished — by wording or by vote, and in which round). A plain `reason` run gains none
of these, and both formats always carry exactly the same triples.
