"""Core data types for Open-NeoVax."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Candidate:
    """A candidate neo-epitope to be evaluated by the pipeline.

    Required attributes:
        candidate_id   – unique identifier
        peptide_wt     – wild-type (reference) sequence
        peptide_mut    – mutated (neo-epitope) sequence
        mut_pos_1based – mutation position (1-indexed)

    Optional attributes:
        gene        – source gene name
        hla_allele  – target HLA allele (e.g. "HLA-A*02:01")
        note        – free-text comment
        meta        – extra metadata
        scores      – scores computed by modules
    """

    candidate_id: str
    peptide_wt: str
    peptide_mut: str
    mut_pos_1based: int
    gene: str = ""
    hla_allele: str = ""
    note: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
