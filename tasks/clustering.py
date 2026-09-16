# from concurrent.futures import ProcessPoolExecutor
def cluster_domain(domain, embeddings, k):
    from sklearn.cluster import KMeans

    kmean_model = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmean_model.fit_predict(embeddings)

    return {
        'labels': labels,
        # 'domain': domain,
        # 'k': k,
        # 'embeddings': embeddings,
        # 'model': kmean_model,
    }

def expose_domain_mean_embedding():
    from tasks.clustering import cluster_domain
    from engine.Util import join, write_json, read_json, CACHED_KNOWLEDGE_BASE_JSON_PATH, CACHED_DOMAIN_K_CLUSTER_PATH
    from typing import cast, List
    import numpy as np

    data = read_json(CACHED_KNOWLEDGE_BASE_JSON_PATH)
    k_domains = read_json(CACHED_DOMAIN_K_CLUSTER_PATH)
    results = {}
    _mean_embeddings_domain = {}

    for domain, n_clusters in k_domains.items():
        cluster_result = cluster_domain(domain, data[domain], n_clusters)
        labels = cast(List[int], cluster_result.get("labels"))

        if domain not in results:
            results[domain] = []

        if domain not in _mean_embeddings_domain:
            _mean_embeddings_domain[domain] = {}

        _embeddings_labels = {}
        
        for index, embedding in enumerate(data[domain]):
            label_name = int(labels[index])
            if label_name not in _embeddings_labels:
                _embeddings_labels[label_name] = []

            if label_name not in _mean_embeddings_domain[domain]:
                _mean_embeddings_domain[domain][label_name] = []
            results[domain].append({
                'embedding': embedding,
                'clustered_label': label_name
            })
            _embeddings_labels[label_name].append(embedding)
        
        for label in labels:
            _mean_embeddings_domain[domain][label] = np.mean(np.array(_embeddings_labels[label]), axis=0).tolist()

    write_json(str(join(
        "knowledge_directory",
        ".cached",
        "domain_mean_embedding.json"
    )),_mean_embeddings_domain)
