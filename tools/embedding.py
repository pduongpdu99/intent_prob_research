from typing import List
from pathlib import Path
from sentence_transformers import SentenceTransformer


MODEL_PATH = Path("weights/ve2")
if not MODEL_PATH.exists():
    MODEL_PATH = "AITeamVN/Vietnamese_Embedding_v2"
    model = SentenceTransformer(MODEL_PATH, model_kwargs={"use_safetensors": True})
else:
    model = SentenceTransformer(str(MODEL_PATH), model_kwargs={"use_safetensors": True})

def sentence_embedding(sentences: List[str]):
    return model.encode(
        sentences,
        convert_to_tensor=True,
    )

def get_similarity_sentence(
    sentence: str,
    sentences: List[str],
):
    se1 = sentence_embedding([sentence])
    sen = sentence_embedding(sentences)

    return model.similarity(sen, se1)


def get_similarity_embeddings(
    embedding,
    db_embeddings,
):
    return model.similarity(
        embedding,
        db_embeddings,
    )

def knowledge_directory_embedding():
    from engine.Util import (
        KNOWLEDGE_BASE_JSON_PATH,
        CACHED_KNOWLEDGE_BASE_JSON_PATH,
        CACHED_KLB_EMBEDDING_JSON_PATH,
        CACHED_KNOWLEDGE_BASE_JSON_PATH,
        CACHED_KLB_EMBEDDING_JSON_PATH,
        read_json
    )
    import json
    import numpy as np
    from sklearn.preprocessing import normalize

    knowledge_base_json = read_json(KNOWLEDGE_BASE_JSON_PATH) or {}

    results = {}
    averages = {}

    print("We have ", knowledge_base_json.keys())
    for key in knowledge_base_json.keys():
        print("STARTING WITH ", key)
        results[key] = sentence_embedding(knowledge_base_json[key]).tolist()
        averages[key] = np.mean(results[key], axis=0)
        averages[key] = normalize(averages[key].reshape(1, -1))[0].tolist()

        print("ENDED Embedding ", key,"\n")

    _file = Path(CACHED_KNOWLEDGE_BASE_JSON_PATH)
    _file.parent.mkdir(parents=True, exist_ok=True)
    _file.touch(exist_ok=True)

    _file = Path(CACHED_KLB_EMBEDDING_JSON_PATH)
    _file.parent.mkdir(parents=True, exist_ok=True)
    _file.touch(exist_ok=True)

    with open(CACHED_KNOWLEDGE_BASE_JSON_PATH, "w", encoding="utf-8") as file:
        file.write(json.dumps(results, indent=1))

    with open(CACHED_KLB_EMBEDDING_JSON_PATH, "w", encoding="utf-8") as file:
        file.write(json.dumps(averages, indent=1))

def get_nearest_index(
    embedding,
    target_embeddings,
):
    scores = get_similarity_embeddings(embedding, target_embeddings).tolist()
    index = -1
    _max = 0
    for i in range(0, len(scores[0])):
        score = scores[0][i]
        if _max >= score: continue
        _max = score
        index = i

    return _max, index
    
def find_nearest_label(prompt: str):
    from engine.Util import read_json, CACHED_KLB_EMBEDDING_JSON_PATH
    embedding = sentence_embedding([prompt]).tolist()
    klb_dict = read_json(CACHED_KLB_EMBEDDING_JSON_PATH)
    klb_labels = list(klb_dict.keys())
    klb_embeddings = list(klb_dict.values())
    nearest_value, nearest_index = get_nearest_index(embedding, klb_embeddings)
    return {
        "index": nearest_index,
        "value": nearest_value,
        "label": klb_labels[nearest_index]
    }

def expose_KLB_empty(state: dict = {}):
    from pathlib import Path
    from typing import cast
    from engine.Util import (
        write_json, 
        read_json,
        CACHED_KNOWLEDGE_BASE_WITH_EMPTY_PATH,
        CACHED_KNOWLEDGE_BASE_JSON_PATH
    )
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

def expose_elbow(state: dict = {}):
    from pathlib import Path
    import numpy as np
    from engine.Util import (
        write_json, 
        read_json,
        elbow_method,
        CACHED_KNOWLEDGE_BASE_ELBOW_PATH,
    )
    if 'data' not in state:
        raise ValueError("Data empty")

    if 'domains' not in state:
        raise ValueError("Domain empty")
        
    data = state['data']
    domains = state['domains']
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

def expose_k_optimize(state: dict = {}):
    from engine.Util import (
        write_json, 
        find_elbow,
        read_json,
        CACHED_DOMAIN_K_CLUSTER_PATH,
    )
    collection = state['elbow']
    columns = state['domains']

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