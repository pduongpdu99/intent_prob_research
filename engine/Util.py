import os
import unicodedata
from dotenv import load_dotenv
import numpy as np
from typing import Any
from pathlib import Path

load_dotenv()
HF_TOKEN = os.getenv("hf_token")

root_directory_path = Path.cwd()

def join(*sub:str):
    return os.path.join(root_directory_path, *sub)

def remove_accent(vietnamese_text: str):
    nfd = unicodedata.normalize("NFD", vietnamese_text)
    stripped = ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')
    stripped = stripped.replace('Đ', 'D').replace('đ', 'd')
    return unicodedata.normalize('NFC', stripped)

def get_vietnamese_stopwords():
    stop_words = []
    with open(os.path.join(root_directory_path, KNOWLEDGE_DIRECTORY, "stop_words.txt"), "r") as f:
        stop_words = [i.strip() for i in f.readlines() if i.strip() != ""]

    return stop_words

def get_non_sw(kv_structure=False):
    result = []
    with open(os.path.join(root_directory_path, KNOWLEDGE_DIRECTORY, "non_stop_words.txt"), "r") as f:
        result = [i.strip() for i in f.readlines() if i.strip() != ""]

    if kv_structure:
        r = {}
        for i in result:
            first_word = i.split(" ")[0]
            if first_word not in r:
                r[first_word] = []
            r[first_word].append(i)
        return r

    return result

def read_json(path: str) -> dict:
    import json
    try :
        with open(path, "r", encoding="utf-8") as file:
            _ = file.read().strip()
            if len(_) == 0: _ = '{}'
            return json.loads(_)
    except FileNotFoundError as e:
        print(e)
        return {}

def write_json(path: str, data: Any) -> None:
    import json
    try :
        with open(path, "w", encoding="utf-8") as file:
            file.write(json.dumps(data, indent=1))
            print("Write completed")
    except FileNotFoundError as e:
        print(e)
        return None

def flat2(arr_2d):
    return [col for row in arr_2d for col in row]

def elbow_method(X: np.ndarray, kmax: int=10):
    from sklearn.cluster import KMeans
    from scipy.spatial.distance import cdist
    k_range = range(1, kmax+1)

    distorions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}

    for k in k_range:
        kmean_model = KMeans(n_clusters=k, random_state=42).fit(X)
        distorions.append(float(sum(np.min(cdist(X, kmean_model.cluster_centers_, "euclidean"),axis=1)**2)/X.shape[0]))
        inertias.append(kmean_model.inertia_)
        mapping1[k] = distorions[-1]
        mapping2[k] = inertias[-1]
    return distorions,inertias,mapping1,mapping2

def find_elbow(ks, ws):
    A = np.array([ks[0], ws[0]], dtype=float)
    B = np.array([ks[-1], ws[-1]], dtype=float)
    AB = A-B
    AB_norm = np.linalg.norm(AB)


    distances = []
    for k, w in zip(ks,ws):
        P = np.array([k, w],dtype=float)
        AP = A-P
        cross = AB[0] * AP[1] - AB[1] * AP[0]
        distance = abs(cross) / AB_norm
        distances.append(distance)

    elbow_index = np.argmax(distances)
    return ks[elbow_index], distances

def expose_clustering_model(domain: str):
    from sklearn.cluster import KMeans
    clusters = read_json(CACHED_DOMAIN_K_CLUSTER_PATH)
    embeddings_domains: dict = read_json(CACHED_KNOWLEDGE_BASE_WITH_EMPTY_PATH)
    X = embeddings_domains[domain]
    kmean_model = KMeans(n_clusters=clusters[domain]).fit(X)
    return kmean_model

def update_clustering_model(model, embeddings):
    model.predict(embeddings)
    return model


# PATH
DEFAULT_TEMPLATE_NAME = "required_template"

KNOWLEDGE_DIRECTORY = join("knowledge_directory")

KNOWLEDGE_BASE_TXT_PATH = join(KNOWLEDGE_DIRECTORY, "knowledge_base.txt")
REQUIRED_TEMPLATE_PATH = join(KNOWLEDGE_DIRECTORY, "required_template.txt")

DATA_COLLECTION_XLSX_PATH = join(KNOWLEDGE_DIRECTORY, "data_collection.xlsx")

KNOWLEDGE_BASE_JSON_PATH = join(KNOWLEDGE_DIRECTORY, ".cached", "knowledge_base.json")
KLB_EMBEDDING_JSON_PATH = join(KNOWLEDGE_DIRECTORY, "klb_embedding_average.json")
ENTITIES_PATH = join(KNOWLEDGE_DIRECTORY, ".cached", DEFAULT_TEMPLATE_NAME,"entities.json")
TRIGGERS_PATH = join(KNOWLEDGE_DIRECTORY, ".cached", DEFAULT_TEMPLATE_NAME,"triggers.json")
RELATION_PATH = join(KNOWLEDGE_DIRECTORY, ".cached", DEFAULT_TEMPLATE_NAME,"relation.json")
CACHED_KNOWLEDGE_BASE_JSON_PATH = join(KNOWLEDGE_DIRECTORY, ".cached", "knowledge_base","knowledge_base.json")
CACHED_DOMAIN_K_CLUSTER_PATH = join(KNOWLEDGE_DIRECTORY, ".cached", "knowledge_base","domain_k_cluster.json")
CACHED_KNOWLEDGE_BASE_WITH_EMPTY_PATH = join(KNOWLEDGE_DIRECTORY, ".cached", "knowledge_base","knowledge_base_with_empty.json")
CACHED_KNOWLEDGE_BASE_ELBOW_PATH = join(KNOWLEDGE_DIRECTORY, ".cached", "knowledge_base","knowledge_base_elbow_cache.json")
CACHED_KLB_EMBEDDING_JSON_PATH = join(KNOWLEDGE_DIRECTORY, ".cached", "knowledge_base","klb_embedding_average.json")

SOFTWARE_ROLE_PATH = join(KNOWLEDGE_DIRECTORY, "roles", "software-data.json")
SOFTWARE_DOMAIN_PATH = join(KNOWLEDGE_DIRECTORY, "domain", "software-domain.json")