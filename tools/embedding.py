from typing import List
from pathlib import Path

from sentence_transformers import SentenceTransformer


MODEL_PATH = Path("weights/ve2")

model = SentenceTransformer(str(MODEL_PATH))

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