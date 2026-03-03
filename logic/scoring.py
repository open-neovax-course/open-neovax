"""Score normalization and aggregation for candidates."""

from __future__ import annotations

from logic.types import Candidate


def normalize_scores(candidates: list[Candidate]) -> list[Candidate]:
    """Normalize each score between 0 and 1 (min-max per score name).

    - If min == max for a given score, all candidates receive 0.5
    - Missing scores for some candidates are replaced by 0.0
    """
    if not candidates:
        return candidates

    # Collect all existing score names
    all_score_names: set[str] = set()
    for c in candidates:
        all_score_names.update(c.scores.keys())

    # Exclude total_score from a previous normalization
    all_score_names.discard("total_score")

    for name in all_score_names:
        values = [c.scores.get(name, 0.0) for c in candidates]
        min_val = min(values)
        max_val = max(values)

        for c in candidates:
            raw = c.scores.get(name, 0.0)
            if max_val == min_val:
                c.scores[name] = 0.5
            else:
                c.scores[name] = (raw - min_val) / (max_val - min_val)

    return candidates


def compute_total_scores(
    candidates: list[Candidate],
    weights: dict[str, float] | None = None,
) -> list[Candidate]:
    """Compute the weighted total score for each candidate.

    If weights is None, all scores have equal weight (1/N).
    The result is stored in candidate.scores["total_score"].
    Candidates are sorted by descending total score.
    """
    if not candidates:
        return candidates

    # Score names (excluding total_score)
    all_score_names: set[str] = set()
    for c in candidates:
        all_score_names.update(c.scores.keys())
    all_score_names.discard("total_score")

    if not all_score_names:
        for c in candidates:
            c.scores["total_score"] = 0.0
        return candidates

    # Default weights: equal
    if weights is None:
        n = len(all_score_names)
        weights = {name: 1.0 / n for name in all_score_names}

    for c in candidates:
        total = 0.0
        for name, w in weights.items():
            total += c.scores.get(name, 0.0) * w
        c.scores["total_score"] = total

    # Sort descending by total_score
    candidates.sort(key=lambda c: c.scores["total_score"], reverse=True)

    return candidates


def aggregate(
    candidates: list[Candidate],
    weights: dict[str, float] | None = None,
) -> list[Candidate]:
    """Normalize, compute totals and sort. Convenience function."""
    candidates = normalize_scores(candidates)
    candidates = compute_total_scores(candidates, weights)
    return candidates
