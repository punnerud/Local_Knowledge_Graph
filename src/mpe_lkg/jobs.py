"""Runs that outlive the request that started them, so RDF can be fetched headless.

The browser holds one SSE connection open for the length of a run, which suits a
person watching and suits nothing else. A script wants to start a run, get an
identifier back immediately, poll it, take the RDF, and be able to stop it when
the answer stopped being worth waiting for.

Three properties this has to get right, because each is a way background work
usually goes wrong:

* **Killing actually stops the work.** A cancel flag that is only read after the
  loop finishes has cancelled nothing. The generator is checked between events,
  which is where a reasoning run spends its time, so a kill lands within one step
  rather than at the end.
* **Finishing is observable.** ``done`` is set in a ``finally``, so a run that
  raised is finished-with-an-error rather than eternally "running" -- the state a
  poller can wait on forever.
* **It does not grow without bound.** Completed jobs are kept for a while and then
  dropped, oldest first. A long-lived server that keeps every run it ever did is
  a leak with a nice API in front of it.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Iterator

# How many finished runs to keep. Enough that a poller which went away for a cup
# of tea still finds its result; small enough that a server left up for a month
# does not hold a month of transcripts.
MAX_KEPT = 64


class Job:
    """One reasoning run, its events, and whether anyone still wants it."""

    def __init__(self, question: str) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.question = question
        # How the run was asked for: reason, explore or settle. Reported so a
        # poller knows what it is waiting for -- a settle run is minutes, not
        # seconds, and a caller that cannot tell them apart will time out on one.
        self.mode = "reason"
        self.session = ""
        self.created = time.time()
        self.finished: float | None = None
        self.events: list[dict] = []
        self.answer = ""
        self.error = ""
        # Set by kill(); read between events by the runner.
        self.cancelled = threading.Event()
        self.done = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    @property
    def state(self) -> str:
        if not self.done.is_set():
            return "cancelling" if self.cancelled.is_set() else "running"
        if self.error:
            return "error"
        return "cancelled" if self.cancelled.is_set() else "done"

    def append(self, event: dict) -> None:
        with self._lock:
            self.events.append(event)

    def snapshot(self) -> list[dict]:
        with self._lock:
            return list(self.events)

    def of_type(self, *kinds: str) -> list[dict]:
        return [e for e in self.snapshot() if e.get("type") in kinds]

    def graph(self) -> dict:
        """The most recent graph payload seen, which is the complete one."""
        for event in reversed(self.snapshot()):
            if event.get("graph"):
                return event["graph"]
        return {"nodes": [], "edges": []}

    def status(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "mode": self.mode,
            "session": self.session,
            "state": self.state,
            "steps": len(self.of_type("step")),
            "events": len(self.snapshot()),
            "answer": self.answer,
            "error": self.error,
            "created": self.created,
            "finished": self.finished,
            "seconds": round((self.finished or time.time()) - self.created, 1),
        }

    def kill(self) -> None:
        self.cancelled.set()

    def start(self, events: Iterator[dict]) -> None:
        self._thread = threading.Thread(target=self._run, args=(events,), daemon=True)
        self._thread.start()

    def _run(self, events: Iterator[dict]) -> None:
        try:
            for event in events:
                self.append(event)
                if event.get("type") == "final":
                    self.answer = event.get("content", "")
                elif event.get("type") == "error":
                    self.error = event.get("message", "error")
                # Between events, which is where the time goes. A flag read only
                # after the loop would cancel nothing.
                if self.cancelled.is_set():
                    break
        except Exception as exc:  # noqa: BLE001 -- a crashed run is a state, not a stop
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            # In finally on purpose: a run that raised must become finished, or a
            # poller waits on "running" forever.
            self.finished = time.time()
            self.done.set()


class Registry:
    """Jobs by id, oldest finished ones dropped."""

    def __init__(self, max_kept: int = MAX_KEPT) -> None:
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_kept

    def add(self, job: Job) -> Job:
        with self._lock:
            self._jobs[job.id] = job
            self._evict()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def _evict(self) -> None:
        # Only finished jobs are ever dropped: evicting a running one would leave
        # a thread writing into an object nobody can reach.
        finished = [j for j in self._jobs.values() if j.done.is_set()]
        while len(self._jobs) > self._max and finished:
            self._jobs.pop(finished.pop(0).id, None)
