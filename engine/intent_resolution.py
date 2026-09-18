import re
from collections.abc import Mapping, Sequence


WORD_PATTERN = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)
CONFIDENCE_THRESHOLD = 0.75
MARGIN_THRESHOLD = 0.05


def _terms(value: str) -> set[str]:
    return {
        term.casefold()
        for term in WORD_PATTERN.findall(value)
        if len(term) >= 2
    }


def _candidate_parts(candidate: object) -> tuple[str, list[str], str | None]:
    if isinstance(candidate, str):
        return candidate, [], None
    if not isinstance(candidate, Mapping):
        raise TypeError("Each candidate intent must be a string or mapping")

    name = candidate.get("intent") or candidate.get("code") or candidate.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Candidate intent must contain intent, code, or name")
    examples = candidate.get("examples", [])
    if isinstance(examples, str):
        examples = [examples]
    if not isinstance(examples, Sequence):
        raise ValueError(f"Examples for {name} must be a sequence")
    domain = candidate.get("domain")
    return name, [str(example) for example in examples], str(domain) if domain else None


def _overlap(left: set[str], right: set[str]) -> float:
    if not left:
        return 0.0
    return len(left & right) / len(left)


def _signal_terms(signals: Sequence[Mapping[str, str]]) -> set[str]:
    terms = set()
    for signal in signals:
        terms.update(_terms(str(signal.get("text", ""))))
        terms.update(_terms(str(signal.get("type", ""))))
    return terms


def _slot_value(signal: Mapping[str, str]) -> str:
    return str(signal.get("text", ""))


def _score_candidate(
    original_prompt: str,
    enriched_prompt: str,
    selected_domain: str,
    entities: Sequence[Mapping[str, str]],
    triggers: Sequence[Mapping[str, str]],
    candidate: object,
) -> dict:
    name, examples, candidate_domain = _candidate_parts(candidate)
    prompt_terms = _terms(f"{original_prompt} {enriched_prompt}")
    example_terms = (
        set().union(*(_terms(example) for example in examples))
        if examples
        else set()
    )
    entity_terms = _signal_terms(entities)
    trigger_terms = _signal_terms(triggers)

    semantic_score = _overlap(prompt_terms, example_terms)
    entity_score = _overlap(entity_terms, example_terms)
    trigger_score = _overlap(trigger_terms, example_terms)
    domain_score = 1.0 if candidate_domain == selected_domain else 0.0
    if candidate_domain is None:
        domain_score = _overlap(_terms(selected_domain), example_terms)

    score = (
        0.45 * semantic_score
        + 0.20 * domain_score
        + 0.20 * trigger_score
        + 0.15 * entity_score
    )
    return {
        "intent": name,
        "score": round(score, 4),
        "semantic_score": round(semantic_score, 4),
        "domain_score": round(domain_score, 4),
        "trigger_score": round(trigger_score, 4),
        "entity_score": round(entity_score, 4),
        "examples": examples[:3],
    }


def _fill_slots(
    entities: Sequence[Mapping[str, str]],
    triggers: Sequence[Mapping[str, str]],
) -> dict[str, str]:
    slots: dict[str, str] = {}
    for entity in entities:
        entity_type = str(entity.get("type", "")).lower()
        text = _slot_value(entity)
        if not text:
            continue
        slot_name = {
            "sys": "system",
            "feature": "feature",
            "tech": "technology",
            "data": "data",
            "security": "security_requirement",
            "constraint": "constraint",
            "org": "actor",
        }.get(entity_type, entity_type or "entity")
        slots.setdefault(slot_name, text)

    if triggers:
        slots["action"] = _slot_value(triggers[0])
    return slots


def resolve_intent(
    *,
    original_prompt: str,
    enriched_prompt: str,
    selected_domain: str,
    entities: Sequence[Mapping[str, str]],
    triggers: Sequence[Mapping[str, str]],
    candidate_intents: Sequence[object],
) -> dict:
    """Resolve an intent without creating labels outside candidate_intents."""
    if not candidate_intents:
        return {
            "selected_domain": selected_domain,
            "intent": None,
            "confidence": 0.0,
            "reasoning": "Không có candidate intent hợp lệ để đối chiếu.",
            "slots": _fill_slots(entities, triggers),
            "requires_clarification": True,
            "clarification_prompt": (
                "Vui lòng cung cấp candidate intents hợp lệ "
                "cho domain này."
            ),
        }

    scored = sorted(
        (
            _score_candidate(
                original_prompt,
                enriched_prompt,
                selected_domain,
                entities,
                triggers,
                candidate,
            )
            for candidate in candidate_intents
        ),
        key=lambda item: item["score"],
        reverse=True,
    )
    best = scored[0]
    next_score = scored[1]["score"] if len(scored) > 1 else 0.0
    margin = round(best["score"] - next_score, 4)
    confidence = round(min(1.0, best["score"]), 2)
    ambiguous = margin < MARGIN_THRESHOLD
    requires_clarification = confidence < CONFIDENCE_THRESHOLD or ambiguous

    trigger_text = ", ".join(
        str(item.get("text", ""))
        for item in triggers
        if item.get("text")
    )
    entity_text = ", ".join(
        str(item.get("text", ""))
        for item in entities
        if item.get("text")
    )
    reasoning = (
        f"Intent {best['intent']} phù hợp nhất với domain {selected_domain}; "
        f"trigger [{trigger_text or 'không có'}] "
        f"và entity [{entity_text or 'không có'}] "
        f"cho điểm kết hợp {confidence:.2f}."
    )

    return {
        "selected_domain": selected_domain,
        "intent": best["intent"],
        "confidence": confidence,
        "reasoning": reasoning,
        "slots": _fill_slots(entities, triggers),
        "requires_clarification": requires_clarification,
        "clarification_prompt": (
            "Bạn có thể làm rõ mục tiêu hoặc chọn một trong các intent "
            "ứng viên phù hợp không?"
            if requires_clarification
            else None
        ),
    }
