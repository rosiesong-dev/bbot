from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedding_model = OpenAIEmbeddings()


semantic_cache_store = []


def get_embedding(text):
    return embedding_model.embed_query(text)


def search_semantic_cache(query, threshold=0.92):
    query_embedding = get_embedding(query)

    best_score = 0
    best_result = None

    for item in semantic_cache_store:
        score = cosine_similarity(
            [query_embedding],
            [item["embedding"]]
        )[0][0]

        if score > best_score:
            best_score = score
            best_result = item

    if best_score >= threshold:
        print(f"⚡ Semantic Cache Hit! score={best_score:.4f}")
        return best_result["data"]

    return None


def save_semantic_cache(query, data):
    semantic_cache_store.append({
        "query": query,
        "embedding": get_embedding(query),
        "data": data
    })

    print("💾 Semantic Cache Saved!")