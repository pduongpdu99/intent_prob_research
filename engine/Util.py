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

def read_json(path: str) -> dict | None:
    import json
    try :
        with open(path, "r", encoding="utf-8") as file:
            _ = file.read().strip()
            if len(_) == 0: _ = '{}'
            return json.loads(_)
    except FileNotFoundError as e:
        print(e)
        return None

def flat2(arr_2d):
    return [col for row in arr_2d for col in row]

# PATH
DEFAULT_TEMPLATE_NAME = "required_template"

KNOWLEDGE_DIRECTORY = join("knowledge_directory")

KNOWLEDGE_BASE_TXT_PATH = join(KNOWLEDGE_DIRECTORY, "knowledge_base.txt")
REQUIRED_TEMPLATE_PATH = join(KNOWLEDGE_DIRECTORY, "required_template.txt")

DATA_COLLECTION_XLSX_PATH = join(KNOWLEDGE_DIRECTORY, "data_collection.xlsx")

KNOWLEDGE_BASE_JSON_PATH = join(KNOWLEDGE_DIRECTORY, "knowledge_base.json")
ENTITIES_PATH = join(KNOWLEDGE_DIRECTORY, ".cached", DEFAULT_TEMPLATE_NAME,"entities.json")
TRIGGERS_PATH = join(KNOWLEDGE_DIRECTORY, ".cached", DEFAULT_TEMPLATE_NAME,"triggers.json")
RELATION_PATH = join(KNOWLEDGE_DIRECTORY, ".cached", DEFAULT_TEMPLATE_NAME,"relation.json")
SOFTWARE_ROLE_PATH = join(KNOWLEDGE_DIRECTORY, "roles", "software-data.json")
SOFTWARE_DOMAIN_PATH = join(KNOWLEDGE_DIRECTORY, "domain", "software-domain.json")