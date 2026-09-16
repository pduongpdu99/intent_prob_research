from engine.enums import StateKeys

def __expose_KLB_empty():
    from pathlib import Path
    from typing import cast, List
    from engine.Util import (
        write_json, 
        read_json,
        CACHED_KNOWLEDGE_BASE_WITH_EMPTY_PATH,
        CACHED_KNOWLEDGE_BASE_JSON_PATH
    )
    from tasks.embedding import sentence_embedding
    data = {}
    columns = []
    if not Path(CACHED_KNOWLEDGE_BASE_WITH_EMPTY_PATH).exists():
        data = read_json(CACHED_KNOWLEDGE_BASE_JSON_PATH)
        columns = list(data.keys())
        _max = max([len(i) for i in data.values()])
        for col in columns:
            t = cast(List[str], data.get(col))
            t = t + list(sentence_embedding(([''] * (_max - len(t))) ).tolist())
            data.update({col: t})
        write_json(CACHED_KNOWLEDGE_BASE_WITH_EMPTY_PATH, data)
    else:
        data = read_json(CACHED_KNOWLEDGE_BASE_WITH_EMPTY_PATH)
        columns = list(data.keys())
    return columns, data

def __expose_KLB():
    from engine.Util import (
        read_json,
        CACHED_KNOWLEDGE_BASE_JSON_PATH
    )
    data = read_json(CACHED_KNOWLEDGE_BASE_JSON_PATH)
    columns = list(data.keys())
    return columns, data

def __expose_elbow(state: dict = {}):
    from pathlib import Path
    import numpy as np
    from engine.Util import (
        write_json, 
        read_json,
        elbow_method,
        CACHED_KNOWLEDGE_BASE_ELBOW_PATH,
    )
    if StateKeys.DATA.value not in state:
        raise ValueError("Data empty")

    if StateKeys.DOMAINS.value not in state:
        raise ValueError("Domain empty")
        
    data = state[StateKeys.DATA.value]
    domains = state[StateKeys.DOMAINS.value]
    collection = []
    if not Path(CACHED_KNOWLEDGE_BASE_ELBOW_PATH).exists():
        for domain in domains:
            X = data[domain]

            distorions, inertias, mapping1, mapping2 = elbow_method(np.array(X),10)
            collection.append({
                "distorions": distorions,
                "inertias": inertias,
                "mapping1": mapping1,
                "mapping2": mapping2
            })
        write_json(CACHED_KNOWLEDGE_BASE_ELBOW_PATH, collection)
    else:
        collection = read_json(CACHED_KNOWLEDGE_BASE_ELBOW_PATH)

    return collection

def __expose_k_optimize(state: dict = {}):
    from pathlib import Path
    from engine.Util import (
        write_json, 
        find_elbow,
        read_json,
        CACHED_DOMAIN_K_CLUSTER_PATH,
    )
    collection = state[StateKeys.ELBOW.value]
    columns = state[StateKeys.DOMAINS.value]

    if 'elbow' not in state:
        raise ValueError("elbow is not exist")

    if not Path(CACHED_DOMAIN_K_CLUSTER_PATH).exists():
        k_domains = {}
        for index, col in enumerate(collection):
            distorion = col['distorions']
            labels = [i+1 for i in range(len(distorion))]
            k_domains[columns[index]] = find_elbow(labels, distorion)[0]
        write_json(CACHED_DOMAIN_K_CLUSTER_PATH, k_domains)
    else:
        k_domains = read_json(CACHED_DOMAIN_K_CLUSTER_PATH)

    return k_domains

async def preprocessing(state: dict):
    from engine.kernel import Kernel
    kernel = Kernel()
    return {"kernel": kernel}

def export_template(state: dict):
    kernel = state['kernel']
    kernel.extraction_tool.export_from_template()
    state['template_status'] = 200
    return state

def export_klb(state: dict):
    from tasks.embedding import knowledge_directory_embedding
    kernel = state['kernel']
    kernel.extraction_tool.export_knowledge_base_json()
    knowledge_directory_embedding()
    state['klb_status'] = 200
    return state

def clustering(state:dict = {}):
    columns, data = __expose_KLB()
    state[StateKeys.DOMAINS.value] = columns
    state[StateKeys.DATA.value] = data

    collection = __expose_elbow({'domains': columns,'data': data})
    state[StateKeys.ELBOW.value] = collection

    k_domains = __expose_k_optimize({
        'elbow': collection,
        'domains': columns
    })
    state[StateKeys.K_DOMAIN.value] = k_domains
    return state

def mean_clustered_embedding(state:dict={}):
    from tasks.clustering import expose_domain_mean_embedding
    expose_domain_mean_embedding()
    return state

def sync_node(state:dict):
    return state