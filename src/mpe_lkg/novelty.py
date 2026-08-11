"""Has this step been taken before?

Two detectors, both embedding-based, differing only in *what* is embedded. They
catch different failures and neither catches the other's:

* **The move.** Two steps titled "Alternative Answer Exploration" repeat the same
  manoeuvre while saying different things. Measured on a real transcript, that pair
  ranked 17th of 21 on *content* similarity -- below average -- so content cannot
  see it. On titles it scores 1.000 against a healthy run's worst of 0.747.
* **The restatement.** A step that says again what an earlier step already said.
  The title gives no hint; the content scores 0.926 and up.

Thresholds come from a measured distribution over 24 questions and 153 steps, not
from taste. The median step already sits at 0.846 content similarity to some
earlier step and the 75th percentile at 0.900, so a 0.90 bar flags the
more-similar quarter of ordinary steps: "we need to calculate a percentage"
followed by "the formula is (17/100) x 250" scores 0.9013 and is plain progress.
See docs/development.md and docs/claims/loops.json.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Measured, see the module docstring. The title bar is insensitive -- exactly the
# same steps fire anywhere from 0.95 to 0.999 -- so its precise value hardly
# matters. The content bar does matter, and 0.96 is where inspection said the
# flags stop being progress and start being restatement.
REPEATED_MOVE = 0.95
RESTATEMENT = 0.96


def angle(a: np.ndarray, b: np.ndarray) -> float:
    """The angle between two vectors, accurate near zero.

    ``2 - 2*dot`` and ``acos(dot)`` both lose their precision exactly at dot ~ 1,
    which is where duplicates live -- catastrophic cancellation in the first case.
    This form is stable there. The dot product is still fine for *ranking*; it is
    the value compared against a threshold that has to be computed this way.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(2.0 * np.arctan2(np.linalg.norm(a - b), np.linalg.norm(a + b)))


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine, computed through the stable angle rather than from the dot product."""
    return float(np.cos(angle(a, b)))


@dataclass
class Novelty:
    """What, if anything, this step repeats."""

    repeat_of: int | None = None
    kind: str = ""          # "move" | "restatement" | ""
    score: float = 0.0
    density: float = 0.0    # how worked-over this step's neighbourhood already is
    covered: list[str] = field(default_factory=list)

    @property
    def is_repeat(self) -> bool:
        return self.repeat_of is not None

    def describe(self) -> str:
        if not self.is_repeat:
            return ""
        what = "makes the same move as" if self.kind == "move" else "restates"
        return f"{what} step {self.repeat_of + 1} ({self.score:.3f})"


def assess(
    title_vec: np.ndarray,
    content_vec: np.ndarray,
    title_history: list,
    content_history: list,
    labels: list[str] | None = None,
) -> Novelty:
    """Compare a candidate step against everything already said in this run."""
    if not content_history:
        return Novelty()

    titles = np.asarray(title_history, dtype=np.float32)
    contents = np.asarray(content_history, dtype=np.float32)

    move_scores = [similarity(title_vec, t) for t in titles]
    restate_scores = [similarity(content_vec, c) for c in contents]

    best_move = int(np.argmax(move_scores))
    best_restate = int(np.argmax(restate_scores))

    # Density is reported whether or not this is a repeat: it says how worked-over
    # this region already is, which is what colours the node in the graph.
    density = float(np.mean(sorted(restate_scores, reverse=True)[:3]))
    covered = list(labels or [])

    # The move is checked first. It is the more reliable of the two by a wide
    # margin -- a healthy run's worst title pair is 0.747 against a repeat's 1.000,
    # versus 0.880 against 0.926 for content.
    if move_scores[best_move] >= REPEATED_MOVE:
        return Novelty(best_move, "move", move_scores[best_move], density, covered)
    if restate_scores[best_restate] >= RESTATEMENT:
        return Novelty(best_restate, "restatement", restate_scores[best_restate], density, covered)
    return Novelty(None, "", max(restate_scores), density, covered)
