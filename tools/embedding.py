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

    knowledge_base_json = read_json(KNOWLEDGE_BASE_JSON_PATH) or {}

    results = {}
    averages = {}

    print("We have ", knowledge_base_json.keys())
    for key in knowledge_base_json.keys():
        print("STARTING WITH ", key)
        results[key] = sentence_embedding(knowledge_base_json[key]).tolist()

        count = len(results[key])
        averages[key] = (np.sum(results[key],axis=0)/count).tolist()

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

