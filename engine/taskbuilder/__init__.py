from engine.Util import (
    read_json, 
    ENTITIES_PATH, 
    TRIGGERS_PATH, 
    RELATION_PATH
)
entities = read_json(ENTITIES_PATH)
triggers = read_json(TRIGGERS_PATH)
relation = read_json(RELATION_PATH)

print(entities)