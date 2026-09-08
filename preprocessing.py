async def preprocessing(state: dict):
    from engine.extraction import ExtractionEngine
    from tools.embedding import knowledge_directory_embedding
    ee = ExtractionEngine()
    ee.export_from_template()
    ee.export_knowledge_base_json()
    knowledge_directory_embedding()
