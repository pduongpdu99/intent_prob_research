import re
from collections.abc import Mapping


WORD_PATTERN = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)
MIN_TERM_LENGTH = 2
MAX_EVIDENCE = 3


def _terms(value: str) -> set[str]:
    return {
        term.casefold()
        for term in WORD_PATTERN.findall(value)
        if len(term) >= MIN_TERM_LENGTH
    }


def _evidence_score(prompt_terms: set[str], example: str) -> tuple[float, list[str]]:
    shared_terms = sorted(prompt_terms & _terms(example))
    if not prompt_terms:
        return 0.0, shared_terms
    return round(len(shared_terms) / len(prompt_terms), 4), shared_terms


def _select_evidence(
    prompt: str,
    domain: str | None,
    knowledge_base: Mapping[str, list[str]],
) -> list[dict]:
    if not domain:
        return []

    prompt_terms = _terms(prompt)
    candidates = []
    for example in knowledge_base.get(domain, []):
        score, shared_terms = _evidence_score(prompt_terms, example)
        if shared_terms:
            candidates.append({
                "text": example,
                "score": score,
                "shared_terms": shared_terms,
            })

    candidates.sort(
        key=lambda item: (item["score"], len(item["shared_terms"])),
        reverse=True,
    )
    return candidates[:MAX_EVIDENCE]


def _missing_information(
    entities: list[dict],
    triggers: list[dict],
    domain: str | None,
) -> list[str]:
    missing = []
    entity_types = {entity["type"] for entity in entities}

    if not entities:
        missing.append("đối tượng hoặc hệ thống cần xử lý")
    if not triggers:
        missing.append("hành động hoặc mục tiêu nghiệp vụ")
    if "TECH" not in entity_types:
        missing.append("công nghệ hoặc phương thức tích hợp nếu có")
    if "CONSTRAINT" not in entity_types:
        missing.append("ràng buộc về hiệu năng, bảo mật hoặc khả năng mở rộng nếu có")
    if not domain:
        missing.append("domain nghiệp vụ")

    return missing


def enrich_prompt(
    prompt: str,
    *,
    domain: Mapping[str, object] | None,
    entities: list[dict],
    triggers: list[dict],
    knowledge_base: Mapping[str, list[str]],
) -> dict:
    """Build a deterministic context frame from extracted prompt signals.

    This function does not infer a new domain or subintent. The caller supplies
    the already selected domain, while the knowledge base is used only to find
    lexical evidence from that domain.
    """
    domain_name = (
        str(domain["domain"])
        if domain and domain.get("domain")
        else None
    )
    evidence = _select_evidence(prompt, domain_name, knowledge_base)
    signal_entities = [
        {"text": entity["text"], "type": entity["type"]}
        for entity in entities
    ]
    signal_triggers = [
        {"text": trigger["text"], "type": trigger["type"]}
        for trigger in triggers
    ]
    missing = _missing_information(signal_entities, signal_triggers, domain_name)

    return {
        "status": "enriched" if domain_name and evidence else "needs_more_context",
        "source": "rule_base",
        "context": {
            "domain": domain_name,
            "cluster": domain.get("cluster") if domain else None,
            "entities": signal_entities,
            "triggers": signal_triggers,
        },
        "evidence": evidence,
        "missing_information": missing,
        "rules_applied": [
            "domain_selected_before_enrichment",
            "evidence_must_belong_to_selected_domain",
            "evidence_ranked_by_lexical_overlap",
            "missing_fields_come_from_entity_types_and_triggers",
        ],
    }
