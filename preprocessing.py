async def preprocessing(state: dict):
    from engine.kernel import Kernel
    kernel = Kernel()
    return {"kernel": kernel}

def export_template(state: dict):
    kernel = state['kernel']
    kernel.extraction_tool.export_from_template()
    return {"template_status": 200}

def export_klb(state: dict):
    from tools.embedding import knowledge_directory_embedding
    kernel = state['kernel']
    kernel.extraction_tool.export_knowledge_base_json()
    knowledge_directory_embedding()
    return {"klb_status": 200}

def sync_node(state:dict):
    return state