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