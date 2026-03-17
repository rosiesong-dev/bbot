from config import get_conn
from llm_factory import get_embedding

embedding_model = get_embedding()


def retrieve_pages(question: str, top_k: int = 3):
    print(f"\n🔎 [Book] 질문: {question}")
    q_emb = embedding_model.embed_query(question)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT book_name, page_num, content
                FROM book_eng
                ORDER BY embedding <#> %s::vector
                LIMIT %s
            """, (q_emb, top_k))
            rows = cur.fetchall()

    print(f"📄 책 검색 결과: {len(rows)}개")
    results = []
    for book_name, page_num, content in rows:
        print(f"   📘 [{book_name}] 페이지 {page_num}")
        results.append({"book": book_name, "page": page_num, "content": content})
    return results