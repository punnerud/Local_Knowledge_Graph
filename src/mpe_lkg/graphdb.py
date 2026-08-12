"""The knowledge graph as tables something can query.

The RDF export is the graph as a *document* -- for handing to another tool. This
is the graph as a *database*: the same runs, steps, edges and facts, in mpedb
tables that SQL can join, filter and aggregate across every run a session has
ever made. ``GraphDB(":memory:")`` gives the same schema for a single run's
analysis without touching disk.

What gets recorded is exactly what the RDF serialises, drawn from the same job
events by the same extraction -- one definition of what is worth keeping, two
representations of it. The vote rows carry their tally and what the dispute was
about, because "how was this answer reached" is the query this table exists for.
"""

from __future__ import annotations

import time

import mpedb

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS runs (
        id TEXT PRIMARY KEY,
        session TEXT NOT NULL DEFAULT '',
        question TEXT NOT NULL,
        answer TEXT NOT NULL DEFAULT '',
        mode TEXT NOT NULL DEFAULT 'reason',
        created REAL NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS steps (
        run TEXT NOT NULL, n INTEGER NOT NULL,
        title TEXT NOT NULL DEFAULT '', content TEXT NOT NULL DEFAULT ''
    )""",
    """CREATE TABLE IF NOT EXISTS edges (
        run TEXT NOT NULL, a TEXT NOT NULL, b TEXT NOT NULL, sim REAL NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS facts (
        run TEXT NOT NULL, kind TEXT NOT NULL, statement TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS findings (
        run TEXT NOT NULL, question TEXT NOT NULL, answer TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS votes (
        run TEXT NOT NULL, agree INTEGER NOT NULL, disagree INTEGER NOT NULL,
        about TEXT NOT NULL DEFAULT ''
    )""",
    # reached_by, not "by": BY is a reserved word and mpedb's parser refuses
    # it as a column name where sqlite3 tolerated it. "round" survives both.
    """CREATE TABLE IF NOT EXISTS agreement (
        run TEXT NOT NULL, reached_by TEXT NOT NULL, round INTEGER NOT NULL DEFAULT 0
    )""",
]


class GraphDB:
    def __init__(self, path: str) -> None:
        self.path = path
        self.conn = mpedb.connect(path, check_same_thread=False)
        for statement in SCHEMA:
            self.conn.execute(statement)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def record(self, job) -> None:
        """One finished job into the tables. Idempotent per job id."""
        already = self.conn.execute(
            "SELECT COUNT(*) FROM runs WHERE id = ?", (job.id,)).fetchone()[0]
        if already:
            return
        self.conn.execute(
            "INSERT INTO runs (id, session, question, answer, mode, created)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (job.id, getattr(job, "session", ""), job.question, job.answer,
             getattr(job, "mode", "reason"), getattr(job, "created", time.time())))

        for event in job.of_type("step"):
            self.conn.execute(
                "INSERT INTO steps (run, n, title, content) VALUES (?, ?, ?, ?)",
                (job.id, int(event.get("step") or 0),
                 str(event.get("title", "")), str(event.get("content", ""))))

        for edge in job.graph().get("edges", []):
            if edge.get("from") and edge.get("to"):
                self.conn.execute(
                    "INSERT INTO edges (run, a, b, sim) VALUES (?, ?, ?, ?)",
                    (job.id, str(edge["from"]), str(edge["to"]),
                     float(edge.get("value", 0.0))))

        for event in job.of_type("convert"):
            self.conn.execute(
                "INSERT INTO facts (run, kind, statement) VALUES (?, 'conversion', ?)",
                (job.id, str(event.get("result", ""))))
        for event in job.of_type("calc"):
            self.conn.execute(
                "INSERT INTO facts (run, kind, statement) VALUES (?, 'calculation', ?)",
                (job.id, f"{event.get('expression', '')} = {event.get('value', '')}"))

        # Top-level findings only, as in the RDF: a sub-run's findings describe
        # its own question, and flattening the levels loses which is which.
        for event in job.of_type("finding"):
            if not event.get("level"):
                self.conn.execute(
                    "INSERT INTO findings (run, question, answer) VALUES (?, ?, ?)",
                    (job.id, str(event.get("question", "")),
                     str(event.get("answer", ""))))

        for event in job.of_type("vote"):
            self.conn.execute(
                "INSERT INTO votes (run, agree, disagree, about) VALUES (?, ?, ?, ?)",
                (job.id, int(event.get("agree", 0)), int(event.get("disagree", 0)),
                 str(event.get("about", ""))))

        for event in job.of_type("agreed"):
            self.conn.execute(
                "INSERT INTO agreement (run, reached_by, round) VALUES (?, ?, ?)",
                (job.id, str(event.get("by", "vote")), int(event.get("round") or 0)))
            break  # one agreement finishes a run; a second is noise

        self.conn.commit()

    def query(self, sql: str, params: tuple = ()) -> list[tuple]:
        """Programmatic SQL over the recorded graph. Not exposed over HTTP."""
        return list(self.conn.execute(sql, params))

    def session_stats(self, session: str = "") -> dict:
        """The canned query the sessions endpoint reads: what a session has done."""
        # Two plain queries, not SUM(EXISTS(...)): mpedb answers the correlated
        # form with nothing at all, and a missing row here took the sessions
        # endpoint down with it.
        runs = self.conn.execute(
            "SELECT COUNT(*) FROM runs WHERE session = ?", (session,)).fetchone()[0]
        agreed = self.conn.execute(
            "SELECT COUNT(DISTINCT agreement.run) FROM agreement"
            " JOIN runs ON agreement.run = runs.id WHERE runs.session = ?",
            (session,)).fetchone()[0]
        facts = self.conn.execute(
            "SELECT COUNT(*) FROM facts JOIN runs ON facts.run = runs.id"
            " WHERE runs.session = ?", (session,)).fetchone()[0]
        return {"runs": int(runs or 0), "agreed": int(agreed or 0),
                "exact_facts": int(facts or 0)}
