import json
from engine.extraction.extract_inforamtion import build_relations, detect_entities, detect_triggers, extract_tokens
from engine.Util import (
    flat2, 
    REQUIRED_TEMPLATE_PATH, 
    ENTITIES_PATH, 
    TRIGGERS_PATH, 
    RELATION_PATH, 
    KNOWLEDGE_BASE_TXT_PATH, 
    KNOWLEDGE_BASE_JSON_PATH
)
from typing import List
import re

__all__ =[
    # "ext_from_prompt",
    "ExtractionEngine",
]

# def ext_from_prompt(input_prompt: str):
#     tokens = extract_tokens(
#         document=input_prompt,
#         pass_stop_word=True,
#         ngram=4
#     )

#     # flatten a 2D array into a 1D array
#     tokens = flat2(tokens)
    
#     entities = detect_entities(tokens)
#     triggers = detect_triggers(tokens)

#     relation_rules = build_relations(entities=entities, triggers=triggers)
#     return entities, triggers, relation_rules

class ExtractionEngine:
    entities: List[tuple[str, str]]
    triggers: List[tuple[str, str]]
    relation: List[dict]
    tokens: List[str]

    # extended func
    extract_tokens = staticmethod(extract_tokens)
    detect_entities = staticmethod(detect_entities)
    detect_triggers = staticmethod(detect_triggers)
    build_relations = staticmethod(build_relations)

    def __init__(self) -> None:
        """The purpose of this component is to identify entities and relations from user prompts.
        While also being capable of learning from data"""
        self.entities = []
        self.triggers = []
        self.relation = []
        self.tokens = []

        # learning

    def learn(self, another_template: str, is_init=False):
        """In addition to the available data, further learning can take place after initialization"""
        __token = extract_tokens(another_template, pass_stop_word=True)
        self.tokens = list(set(self.tokens + __token))

        __entities = self.detect_entities(self.tokens)
        __triggers = self.detect_triggers(self.tokens)
        __relation = self.build_relations(__entities, __triggers)

        if not is_init:
            self.entities += __entities
            self.triggers += __triggers
            self.relation += __relation
        else:
            self.entities = __entities
            self.triggers = __triggers
            self.relation = __relation

        self.entities = list(set(self.entities))
        self.triggers = list(set(self.triggers))

    def to_entities(self, path: str):
        with open(path, "w") as file:
            file.write(json.dumps({k:v for k, v in self.entities}, ensure_ascii=False, indent=4))

    def to_triggers(self, path: str):
        with open(path, "w") as file:
            file.write(json.dumps({k:v for k, v in self.triggers}, ensure_ascii=False, indent=4))

    def to_relation(self, path: str):
        with open(path, "w") as file:
            file.write(json.dumps(self.relation, ensure_ascii=False, indent=4))

    def export_from_template(self):
        docs = []
        with open(REQUIRED_TEMPLATE_PATH) as file:
            docs = file.readlines()

        for doc in docs:
            self.learn(doc)

        self.to_entities(ENTITIES_PATH)
        self.to_triggers(TRIGGERS_PATH)
        self.to_relation(RELATION_PATH)

    def export_knowledge_base_json(self):
        results = {}
        with open(KNOWLEDGE_BASE_TXT_PATH) as file:
            for line in file.readlines():
                _line = re.split(r"(\d+).$", line)[-1].strip()
                _str = re.split(r"(\[[^\]]*\])", _line)
                if len(_str) == 1: continue
                software_type = _str[1][1:-1]
                required_description = _str[2].strip()

                if software_type not in results:
                    results[software_type] = []

                if required_description not in results[software_type]:
                    results[software_type].append(required_description)

        with open(KNOWLEDGE_BASE_JSON_PATH, "w") as file:
            file.write(json.dumps(results, indent=2,ensure_ascii=False).encode("utf-8").decode())
