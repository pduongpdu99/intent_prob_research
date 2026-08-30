from typing import List
from sentence_transformers import SentenceTransformer
import torch
from engine.Util import HF_TOKEN

model_name="AITeamVN/Vietnamese_Embedding_v2"
model = SentenceTransformer(model_name, token=HF_TOKEN)

def sentence_embedding(sentences: List[str]):
    return model.encode(sentences)


def get_similarity_sentence(sentence: str, sentences:List[str]):
    se1 = sentence_embedding([sentence])
    sen = sentence_embedding(sentences)

    return model.similarity(sen, se1)

def get_similarity_embeddings(embedding: torch.Tensor, db_embeddings:List[torch.Tensor]):
    return model.similarity([embedding], db_embeddings)
