"""Shim: the battery moved into the package as ``mpe_lkg.battery``.

The dev scripts and tests import ``battery`` from this directory, and recorded
claims pin the questions the old ``build`` produced for a given seed -- so this
shim preserves the old surface EXACTLY: ``build()`` returns the arithmetic
domain alone, with the raw seed, which is byte-for-byte the battery this file
used to generate. The new domains are reached through ``mpe_lkg.battery.build``
with ``domains=`` or through ``mpe-lkg battery --domains``.
"""

from __future__ import annotations

from mpe_lkg.battery import GENERATORS, SEED, Question, domain_seed, truth_table
from mpe_lkg.battery import build as _package_build


def build(seed: int = SEED, per_group: int = 4) -> list[Question]:
    return _package_build(seed, per_group, domains=("arithmetic",))


def groups() -> list[str]:
    return sorted({q.group for q in build()})


__all__ = ["SEED", "GENERATORS", "Question", "build", "domain_seed",
           "groups", "truth_table"]
