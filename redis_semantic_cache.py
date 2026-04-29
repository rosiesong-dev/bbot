import json
import redis
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embedding_model = OpenAIEmbeddings()

r = redis.Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True
)


def get_embedding(text):
    return embedding_model.embed_query(text)


def save_semantic_cache(query, data):
    embedding = get_embedding(query)

    payload = {
        "query": query,
        "embedding": embedding,
        "data": data
    }

    key = f"semantic:{query}"

    r.set(
        key,
        json.dumps(payload, ensure_ascii=False)
    )

    print("💾 Semantic Cache Saved to Redis!")


def search_semantic_cache(query, threshold=0.85):
    query_embedding = get_embedding(query)

    best_score = 0
    best_result = None

    for key in r.scan_iter("semantic:*"):
        raw = r.get(key)

        if not raw:
            continue

        item = json.loads(raw)

        stored_embedding = item["embedding"]

        score = cosine_similarity(
            [query_embedding],
            [stored_embedding]
        )[0][0]

        if score > best_score:
            best_score = score
            best_result = item

    if best_score >= threshold:
        print(
            f"⚡ Semantic Cache Hit! score={best_score:.4f}"
        )
        return best_result["data"]

    return None