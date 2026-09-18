import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from engine.domain_resolution import fuse_domain_candidates
from engine.enrichment import enrich_prompt
from engine.extraction.extract_inforamtion import (
    detect_entities,
    detect_triggers,
    extract_tokens,
)
from engine.intent_resolution import resolve_intent
from tasks.embedding import sentence_embedding


KNOWLEDGE_DIR = Path(__file__).parent / "knowledge_directory"
DOMAIN_MEAN_FILE = KNOWLEDGE_DIR / ".cached" / "domain_mean_embedding.json"
KNOWLEDGE_BASE_FILE = KNOWLEDGE_DIR / ".cached" / "knowledge_base.json"
ROLE_FILE = KNOWLEDGE_DIR / "roles" / "software-data.json"
MATCH_THRESHOLD = 0.45
MATCH_MARGIN = 0.04


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


@lru_cache(maxsize=1)
def load_domain_means(
    path: Path = DOMAIN_MEAN_FILE,
) -> dict[str, dict[str, list[float]]]:
    """Load the precomputed mean vector for every cluster of every domain."""
    domains = _read_json(path)
    for domain, clusters in domains.items():
        if not isinstance(clusters, dict) or not clusters:
            raise ValueError(f"Domain has no cluster means: {domain}")
        for cluster, vector in clusters.items():
            if not isinstance(vector, list) or not vector:
                raise ValueError(f"Invalid mean vector: {domain}/{cluster}")
    return domains


def load_roles(path: Path = ROLE_FILE) -> dict[str, str]:
    return _read_json(path)


@lru_cache(maxsize=1)
def load_knowledge_base(path: Path = KNOWLEDGE_BASE_FILE) -> dict[str, list[str]]:
    return _read_json(path)


@lru_cache(maxsize=1)
def _cached_cluster_vectors() -> tuple[tuple[dict, ...], np.ndarray]:
    rows = []
    vectors = []
    for domain, clusters in load_domain_means().items():
        for cluster, vector in clusters.items():
            rows.append({"domain": domain, "cluster": cluster})
            vectors.append(vector)
    return tuple(rows), np.asarray(vectors, dtype=np.float32)


def _cosine_scores(prompt_vector, vectors: np.ndarray) -> list[float]:
    prompt_array = (
        prompt_vector.detach().cpu().numpy()
        if hasattr(prompt_vector, "detach")
        else np.asarray(prompt_vector)
    )
    prompt_array = prompt_array.reshape(1, -1)
    vector_norms = np.linalg.norm(vectors, axis=1)
    prompt_norm = np.linalg.norm(prompt_array, axis=1)[0]
    denominator = vector_norms * prompt_norm
    scores = (vectors @ prompt_array[0]) / np.where(denominator == 0, 1, denominator)
    return scores.tolist()


def _rank_domain_clusters(prompt: str) -> list[dict]:
    rows, vectors = _cached_cluster_vectors()
    prompt_vector = sentence_embedding([prompt])
    scores = _cosine_scores(prompt_vector, vectors)
    ranked = [
        {**row, "score": round(float(score), 4)}
        for row, score in zip(rows, scores)
    ]
    return sorted(ranked, key=lambda item: item["score"], reverse=True)


def _mentioned_roles(prompt: str, roles: dict[str, str]) -> list[str]:
    prompt_lower = prompt.casefold()
    return [role for role in roles.values() if role.casefold() in prompt_lower]


def analyze_prompt(
    prompt: str,
    candidate_intents: list[object] | None = None,
) -> dict:
    all_cluster_matches = _rank_domain_clusters(prompt)
    tokens = extract_tokens(prompt, pass_stop_word=True)
    entities = detect_entities(tokens)
    triggers = detect_triggers(tokens)
    entity_items = [
        {"text": text, "type": entity_type}
        for text, entity_type in entities
    ]
    trigger_items = [
        {"text": text, "type": trigger_type}
        for text, trigger_type in triggers
    ]
    knowledge_base = load_knowledge_base()
    domain_candidates = fuse_domain_candidates(
        all_cluster_matches,
        entities=entity_items,
        triggers=trigger_items,
        knowledge_base=knowledge_base,
    )
    best_candidate = domain_candidates[0] if domain_candidates else {}
    best_cluster = next(
        (
            match
            for match in all_cluster_matches
            if match["domain"] == best_candidate.get("domain")
        ),
        {},
    )
    best_match = {**best_candidate, **best_cluster}
    next_score = (
        domain_candidates[1]["fusion_score"]
        if len(domain_candidates) > 1
        else 0.0
    )
    margin = round(best_match.get("fusion_score", 0.0) - next_score, 4)
    confirmed = (
        bool(best_match)
        and best_match["fusion_score"] >= MATCH_THRESHOLD
        and margin >= MATCH_MARGIN
    )
    enrichment = enrich_prompt(
        prompt,
        domain=best_match,
        entities=entity_items,
        triggers=trigger_items,
        knowledge_base=knowledge_base,
    )
    intent_resolution = None
    if candidate_intents is not None:
        intent_resolution = resolve_intent(
            original_prompt=prompt,
            enriched_prompt=json.dumps(enrichment, ensure_ascii=False),
            selected_domain=str(best_match.get("domain", "")),
            entities=entity_items,
            triggers=trigger_items,
            candidate_intents=candidate_intents,
        )

    return {
        "prompt": prompt,
        "domain": {
            **best_match,
            "margin": margin,
            "confirmed": confirmed,
        },
        "domain_candidates": domain_candidates,
        "related_domains": domain_candidates[1:3],
        "roles": _mentioned_roles(prompt, load_roles()),
        "cluster_matches": all_cluster_matches[:5],
        "enrichment": enrichment,
        "intent_resolution": intent_resolution,
        "tokens": tokens,
        "entities": entity_items,
        "triggers": trigger_items,
        "routing": {
            "status": "accepted" if confirmed else "needs_clarification",
            "decision": "route" if confirmed else "ask_for_clarification",
            "strategy": "top_k_independent_evidence_fusion",
        },
    }


def enter_prompt(prompt: str) -> dict:
    result = analyze_prompt(prompt)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]).strip()
    if not prompt:
        prompt = input("Prompt: ").strip()
    if prompt:
        enter_prompt(prompt)
