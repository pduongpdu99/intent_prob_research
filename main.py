import gc
import pandas as pd
import torch

from engine.Util import join
from tools.embedding import (
    sentence_embedding,
    get_similarity_embeddings,
)


def build_prompt(title, overview):
    return (
        f'tôi đang cần build một hệ thống "{title}" '
        f'thông tin chi tiết như sau "{overview}"'
    )


def runtime(prompts):
    sentence = "làm hệ thống quản lý thông tin"

    with torch.inference_mode():
        sembed = sentence_embedding([sentence])

        for i in range(0, len(prompts), 256):
            batch = prompts[i:i + 256]

            embeddings = sentence_embedding(batch)

            xxx = get_similarity_embeddings(
                sembed,
                embeddings,
            )

            # xử lý xxx ở đây
            print(xxx)

            del embeddings
            del xxx


if __name__ == "__main__":
    dataset = pd.read_csv(
        join("data", "_temp.csv")
    )

    prompts = [
        build_prompt(title, overview)
        for title, overview in dataset.itertuples(index=False)
    ]

    runtime(prompts)

    del prompts
    del dataset

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()