import re
from collections import defaultdict
from collections.abc import Mapping, Sequence


WORD_PATTERN = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)


def _terms(value: str) -> set[str]:
    return {
        term.casefold()
        for term in WORD_PATTERN.findall(value)
        if len(term) >= 2
    }


def _feature_terms(
    entities: Sequence[Mapping[str, str]],
    triggers: Sequence[Mapping[str, str]],
) -> set[str]:
    terms: set[str] = set()
    for signal in (*entities, *triggers):
        terms.update(_terms(str(signal.get("text", ""))))
    return terms


def _feature_compatibility(
    feature_terms: set[str],
    examples: Sequence[str],
) -> tuple[float, list[str]]:
    if not feature_terms or not examples:
        return 0.0, []

    best_score = 0.0
    best_terms: list[str] = []
    for example in examples:
        shared_terms = sorted(feature_terms & _terms(example))
        score = len(shared_terms) / len(feature_terms)
        if (score, len(shared_terms)) > (best_score, len(best_terms)):
            best_score = score
            best_terms = shared_terms
    return round(best_score, 4), best_terms


def fuse_domain_candidates(
    cluster_matches: Sequence[Mapping[str, object]],
    *,
    entities: Sequence[Mapping[str, str]],
    triggers: Sequence[Mapping[str, str]],
    knowledge_base: Mapping[str, Sequence[str]],
    limit: int = 3,
) -> list[dict]:
    """Fuse direct semantic, cluster, and entity/trigger evidence.

    All domains represented by the prototype matches receive feature evidence
    before ranking. No domain is selected before this fusion step.
    """
    grouped: dict[str, list[float]] = defaultdict(list)
    for match in cluster_matches:
        domain = str(match["domain"])
        score = match["score"]
        if isinstance(score, (int, float)):
            grouped[domain].append(float(score))

    feature_terms = _feature_terms(entities, triggers)
    candidates: list[dict] = []
    for domain, scores in grouped.items():
        scores.sort(reverse=True)
        cluster_match = scores[0]
        direct_semantic = sum(scores[:3]) / min(3, len(scores))
        feature_score, shared_terms = _feature_compatibility(
            feature_terms, knowledge_base.get(domain, [])
        )
        fusion_score = (
            0.50 * cluster_match
            + 0.30 * direct_semantic
            + 0.20 * feature_score
        )
        candidates.append({
            "domain": domain,
            "fusion_score": round(fusion_score, 4),
            "cluster_match": round(cluster_match, 4),
            "direct_semantic": round(direct_semantic, 4),
            "entity_trigger_compatibility": feature_score,
            "feature_terms": shared_terms,
            "prototype_count": len(scores),
        })

    candidates.sort(key=lambda item: item["fusion_score"], reverse=True)
    return candidates[:limit]
