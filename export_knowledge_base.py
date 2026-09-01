from engine.extraction import ExtractionEngine

ee = ExtractionEngine()

docs = []
with open("./knowledge_directory/required_template.txt") as file:
    docs = file.readlines()

for doc in docs:
    ee.learn(doc)

ee.to_entities("knowledge_directory/entities.json")
ee.to_triggers("knowledge_directory/triggers.json")
ee.to_relation("knowledge_directory/relation.json")

