import json
from engine.extraction.extract_inforamtion import build_relations, detect_entities, detect_triggers, extract_tokens
from engine.Util import flat2
from typing import List

__all__ =[
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

class ExtractionEngine:
    entities: List[tuple[str, str]]
    triggers: List[tuple[str, str]]
    relation: List[dict]
    tokens: List[str]

    def __init__(self) -> None:
        """The purpose of this component is to identify entities and relations from user prompts.
        While also being capable of learning from data"""
        self.entities = []
        self.triggers = []
        self.relation = []
        self.tokens = []

        # learning

    def learn(self, another_template: str):
        """In addition to the available data, further learning can take place after initialization"""
        __token = flat2(extract_tokens(another_template, pass_stop_word=True))
        self.tokens = list(set(self.tokens + __token))

        __entities = detect_entities(self.tokens)
        __triggers = detect_triggers(self.tokens)
        __relation = build_relations(__entities, __triggers)

        self.entities = __entities
        self.triggers = __triggers
        self.relation = __relation

    def to_entities(self, path: str):
        with open(path, "w") as file:
            file.write(json.dumps({k:v for k, v in self.entities}, ensure_ascii=False, indent=4))

    def to_triggers(self, path: str):
        with open(path, "w") as file:
            file.write(json.dumps({k:v for k, v in self.triggers}, ensure_ascii=False, indent=4))

    def to_relation(self, path: str):
        with open(path, "w") as file:
            file.write(json.dumps(self.relation, ensure_ascii=False, indent=4))