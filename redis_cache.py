import redis
import json

redis_client = redis.Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True
)


def get_cached_answer(question):
    result = redis_client.get(question)

    if result:
        return json.loads(result)

    return None


def save_cached_answer(question, answer_data, expire=3600):
    redis_client.setex(
        question,
        expire,
        json.dumps(answer_data, ensure_ascii=False)
    )