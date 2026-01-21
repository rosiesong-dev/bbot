# bbot_book.py
import os
import pdfplumber
import psycopg2
from dotenv import load_dotenv
from openai import OpenAI
from langchain_upstage import UpstageEmbeddings
import streamlit as st



# ============================= 환경 변수 =============================
load_dotenv()

PDF_PATH = "./books/CaseForACreator-Strobel.pdf"

model = OpenAI(
    api_key=os.getenv("UPSTAGE_API_KEY"),
    base_url=os.getenv("UPSTAGE_BASE_URL")
)

embedding_model = UpstageEmbeddings(
    upstage_api_key=os.getenv("UPSTAGE_API_KEY"),
    model="embedding-query"
)



# ============================= DB 연결 함수 =============================
def get_conn():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT")
    )



# ============================= 테이블 존재 확인 =============================
def table_exists():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'book_eng'
                );
            """)
            return cur.fetchone()[0]



# ============================= PDF → DB 저장 (최초 1회) =============================
def create_book_db():
    with get_conn() as conn:
        with conn.cursor() as cur:

            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            cur.execute("""
            CREATE TABLE IF NOT EXISTS book_eng (
                id SERIAL PRIMARY KEY,
                book_name TEXT,
                page_num INT,
                content TEXT,
                embedding vector(4096)
            );
            """)

            book_name = os.path.basename(PDF_PATH).replace(".pdf", "")
            
            # 🔒 이 책이 이미 DB에 있는지 체크
            cur.execute("""
                SELECT COUNT(*) FROM book_eng 
                WHERE book_name = %s
            """, (book_name,))
            
            if cur.fetchone()[0] > 0:
                print(f"📚 '{book_name}' 이미 DB에 있음 → 생성 스킵")
                return

            print(f"[DB] 책 이름: {book_name}")

            with pdfplumber.open(PDF_PATH) as pdf:
                total_pages = len(pdf.pages)
                print(f"[PDF] 총 페이지 수: {total_pages}")
                inserted = 0
                
                for i, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if not text or len(text.strip()) < 50:
                        continue

                    embedding = embedding_model.embed_query(text)

                    cur.execute("""
                        INSERT INTO book_eng (book_name, page_num, content, embedding)
                        VALUES (%s, %s, %s, %s::vector)
                    """, (book_name, i, text, embedding))
                    
                    inserted += 1
                    
                    if i % 10 == 0:
                        print(f"   페이지 {i}/{total_pages} 처리 중...")

            conn.commit()
            print(f"✅ '{book_name}' DB 삽입 완료 (총 {inserted}페이지)")



# ============================= 벡터 검색 =============================
def retrieve_pages(question: str, top_k: int = 3):
    print(f"\n🔎 질문: {question}")

    q_emb = embedding_model.embed_query(question)

    with get_conn() as conn:
        with conn.cursor() as cur:
            print("📡 벡터 검색 실행")
            cur.execute("""
                SELECT book_name, page_num, content
                FROM book_eng
                ORDER BY embedding <#> %s::vector
                LIMIT %s
            """, (q_emb, top_k))

            rows = cur.fetchall()
            print(f"📄 검색 결과 수: {len(rows)}")

    results = []

    for book_name, page_num, content in rows:
        snippet = content[:500].replace("\n", " ")
        print(f"\n   📘 [{book_name}] 페이지 {page_num}")
        print(f"     {snippet}...\n")

        results.append({
            "book": book_name,
            "page": page_num,
            "content": content
        })

    return results



# # ============================= 답변 생성 =============================
# def generate_answer(question: str):
#     docs = retrieve_pages(question)

#     if not docs:
#         return "📘 제공된 책에는 해당 질문과 관련된 내용이 없습니다."

#     lang_inst = "한국어로 답변하세요." if detect_language(question) == "ko" else "Answer in English."

#     context = "\n\n".join(
#         f"[{d['book']} - 페이지 {d['page']}]\n{d['content']}"
#         for d in docs
#     )

#     # 책 이름 그룹핑
#     book_names = list(set([d['book'] for d in docs]))
    
#     if len(book_names) == 1:
#         # 같은 책인 경우
#         book_name = book_names[0]
#         pages = ", ".join(str(d['page']) for d in docs)
#         pages_info = f"📖 **{book_name}** - 페이지 {pages}"
#     else:
#         # 여러 책인 경우
#         pages_info = "\n".join(
#             f"• {d['book']} - 페이지 {d['page']}"
#             for d in docs
#         )

#     system_prompt = f"""
# 당신은 기독교적 세계관과 창조과학 관점에 기반한 전문가입니다.

# 규칙:
# - 반드시 아래 [책 내용]만 사용
# - 추측하거나 외부 지식 사용 금지
# - 🌍 과학적 관점 / 📜 성경적 관점으로 구분하여 설명

# {lang_inst}
# """

#     user_prompt = f"""
# [책 내용]
# {context}

# [질문]
# {question}
# """

#     res = model.chat.completions.create(
#         model="solar-pro2",
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt}
#         ],
#         temperature=0
#     )

#     return res.choices[0].message.content + f"\n\n---\n{pages_info}"



# # ============================= Streamlit UI =============================
# st.set_page_config(page_title="📘 BeBot", page_icon="🤖", layout="wide")

# st.markdown(
#     "<h2 style='text-align:center;'>🤖 BeBot – Let there be light 🤖</h2>",
#     unsafe_allow_html=True
# )

# st.caption("모든 답변은 성경적 관점과 창조과학에 기반합니다.")

# # 사이드바
# with st.sidebar:
#     st.markdown("### 📚 책 정보")
#     book_name = os.path.basename(PDF_PATH).replace(".pdf", "")
#     st.info(f"**현재 책:** {book_name}")
    
#     st.markdown("---")
#     st.markdown("### ⚙️ 설정")
    
#     if st.button("🗑️ 대화 초기화"):
#         st.session_state.messages = []
#         st.rerun()
    
#     st.markdown("---")
#     st.caption("Powered by Upstage Solar & LangGraph")

# # DB 준비 (딱 1번)
# if "db_ready" not in st.session_state:
#     if not table_exists():
#         with st.spinner("📚 책 DB 생성 중 (최초 1회)…"):
#             create_book_db()
#     st.session_state.db_ready = True

# # 채팅 기록
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

# question = st.chat_input("질문을 입력하세요...")

# if question:
#     st.session_state.messages.append({"role": "user", "content": question})
#     with st.chat_message("user"):
#         st.markdown(question)

#     with st.chat_message("assistant"):
#         with st.spinner("Searching …"):
#             answer = generate_answer(question)
#             st.markdown(answer)

#     st.session_state.messages.append({"role": "assistant", "content": answer})