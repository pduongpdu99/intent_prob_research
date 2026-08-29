import os
import unicodedata

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
    with open(os.path.join(root_directory_path, "data", "stop_words.txt"), "rb") as f:
        stop_words = [i.strip() for i in f.readlines() if i.strip() != ""]

    return stop_words