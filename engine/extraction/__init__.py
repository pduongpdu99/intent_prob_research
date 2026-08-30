from engine.extraction.extract_inforamtion import build_relations, detect_entities, detect_triggers, extract_tokens
from engine.Util import flat2

__all__ =[
    # "build_relations",
    # "detect_entities",
    # "detect_triggers",
    # "extract_tokens",
    "ext_from_prompt",
]

def ext_from_prompt(input_prompt: str):
    tokens = extract_tokens(
        document=input_prompt,
        pass_stop_word=True,
        ngram=4
    )

    # flatten a 2D array into a 1D array
    tokens = flat2(tokens)
    
    entities = detect_entities(tokens)
    triggers = detect_triggers(tokens)

    relation_rules = build_relations(entities=entities, triggers=triggers)
    return relation_rules