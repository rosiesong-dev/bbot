from config import get_conn
from llm_factory import get_embedding

embedding_model = get_embedding()


def retrieve_video_segments(question: str, top_k: int = 3):
    print(f"\n🎬 [Video] 질문: {question}")
    q_emb = embedding_model.embed_query(question)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables WHERE table_name = 'video_db'
                );
            """)
            if not cur.fetchone()[0]:
                print("⚠️ video_db 테이블이 없습니다.")
                return []

            cur.execute("""
                SELECT video_id, title, start_time, end_time, url, content
                FROM video_db
                ORDER BY content_embedding <#> %s::vector
                LIMIT %s
            """, (q_emb, top_k))
            rows = cur.fetchall()

    print(f"📄 영상 검색 결과: {len(rows)}개")
    results = []
    for video_id, title, start, end, url, content in rows:
        print(f"   🎬 {title} ({int(start)}s ~ {int(end)}s)")
        results.append({"video_id": video_id, "title": title, "start": start,
                        "end": end, "url": url, "content": content})
    return results