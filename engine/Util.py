import os
import unicodedata
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("hf_token")

root_directory_path = os.getcwd()

def join(*sub:str):
    return os.path.join(root_directory_path, *sub)

def remove_accent(vietnamese_text: str):
    nfd = unicodedata.normalize("NFD", vietnamese_text)
    stripped = ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')
    stripped = stripped.replace('Đ', 'D').replace('đ', 'd')
    return unicodedata.normalize('NFC', stripped)

def get_vietnamese_stopwords():
    stop_words = []
    with open(os.path.join(root_directory_path, "data", "stop_words.txt"), "r") as f:
        stop_words = [i.strip() for i in f.readlines() if i.strip() != ""]

    return stop_words

def get_domain():
    with open(os.path.join(root_directory_path, "data", "domain.txt"), "r") as f:
        return [i.strip() for i in f.readlines() if i.strip() != ""]

def get_non_sw(kv_structure=False):
    result = []
    with open(os.path.join(root_directory_path, "data", "non_stop_words.txt"), "r") as f:
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

def flat2(arr_2d):
    return [col for row in arr_2d for col in row]

# PATH
KNOWLEDGE_DIRECTORY = join("knowledge_directory")
DATA_COLLECTION_XLSX_PATH = join(KNOWLEDGE_DIRECTORY, "data_collection.xlsx")
ENTITIES_PATH = join(KNOWLEDGE_DIRECTORY, "entities.json")
TRIGGERS_PATH = join(KNOWLEDGE_DIRECTORY, "triggers.json")
RELATION_PATH = join(KNOWLEDGE_DIRECTORY, "relation.json")

# JSON
# ENTITIES_DATA = json.loads(ENTITIES_PATH)
# TRIGGERS_DATA = json.loads(TRIGGERS_PATH)
# RELATION_DATA = json.loads(RELATION_PATH)