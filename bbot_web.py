# bbot_web.py
import os
import json
from datetime import datetime
from typing import List, Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv

import psycopg2
from openai import OpenAI
from langchain_upstage import UpstageEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import tiktoken

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from bbot_book import retrieve_pages


# ==================== 환경 변수 로드 ====================
load_dotenv()   
api_key = os.getenv("UPSTAGE_API_KEY")
base_url = os.getenv("UPSTAGE_BASE_URL")


# ==================== 모델 초기화 ====================
model = OpenAI(api_key=api_key, base_url=base_url)
embedding_model = UpstageEmbeddings(
    upstage_api_key=api_key,
    model="embedding-query"
)


# ==================== DB 연결 ====================
def get_conn():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT")
    )


conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    port=os.getenv("DB_PORT")
)


# ==================== DB 관련 함수 ====================
enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

def split_text_by_tokens(text: str, max_tokens: int = 4000):
    words = text.split()
    chunks = []
    chunk = []
    tokens_so_far = 0

    for word in words:
        word_tokens = count_tokens(word + " ")
        if tokens_so_far + word_tokens > max_tokens:
            chunks.append(" ".join(chunk))
            chunk = []
            tokens_so_far = 0
        chunk.append(word)
        tokens_so_far += word_tokens

    if chunk:
        chunks.append(" ".join(chunk))
    return chunks


def create_db(folder_path: str, db_name: str = "bbot_db", max_tokens: int = 4000):
    """extracted_texts 폴더 안 모든 텍스트 파일을 파싱하여 DB에 저장"""
    cur = conn.cursor()
    print("[DB] 연결 성공")

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS crawled_data (  
            id SERIAL PRIMARY KEY,
            title TEXT,
            url TEXT,
            crawl_time TIMESTAMP,
            content TEXT,
            content_embedding vector(4096)
        );
    """)

    failed_files = []
    inserted_count = 0
    files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    print(f"[DB] 폴더에서 {len(files)}개 파일 발견")

    for idx, fname in enumerate(files, start=1):
        try:
            with open(os.path.join(folder_path, fname), "r", encoding="utf-8") as f:
                full_text = f.read()

            lines = full_text.split('\n')
            
            title = ""
            url = ""
            crawl_time = None
            content = ""
            
            content_start_idx = 0
            for i, line in enumerate(lines):
                if line.startswith("Title:"):
                    title = line.replace("Title:", "").strip()
                elif line.startswith("URL:"):
                    url = line.replace("URL:", "").strip()
                elif line.startswith("Crawl Time:"):
                    crawl_time_str = line.replace("Crawl Time:", "").strip()
                    try:
                        crawl_time = datetime.fromisoformat(crawl_time_str.replace('+09:00', ''))
                    except:
                        crawl_time = datetime.now()
                elif line.startswith("Content:"):
                    content_start_idx = i + 1
                    break
            
            if content_start_idx > 0:
                content = '\n'.join(lines[content_start_idx:]).strip()
            else:
                content = full_text
            
            if not title:
                title = fname.replace(".txt", "")
            
            chunks = split_text_by_tokens(content, max_tokens=max_tokens)

            for chunk_idx, chunk in enumerate(chunks):
                embedding_vector = embedding_model.embed_query(chunk)

                cur.execute(
                    "INSERT INTO crawled_data (title, url, crawl_time, content, content_embedding) VALUES (%s, %s, %s, %s, %s::vector)",
                    (title, url, crawl_time, chunk, embedding_vector)
                )
                inserted_count += 1

            if idx % 10 == 0:
                print(f"[DB] {idx}개 파일 처리 완료")
                print(f"   최근 파일: {title[:50]}...")
                print(f"   URL: {url[:80] if url else '(없음)'}")

        except Exception as e:
            print(f"[ERROR] 파일 삽입 실패: {fname}")
            print(f"   에러 내용: {e}")
            failed_files.append(fname)
            conn.rollback()

    conn.commit()

    cur.execute("SELECT COUNT(*) FROM crawled_data;")
    total_count = cur.fetchone()[0]
    print(f"\n[DB] ===== 최종 통계 =====")
    print(f"[DB] 총 데이터 개수: {total_count}")
    print(f"[DB] 성공적으로 삽입된 청크 수: {inserted_count}")
    print(f"[DB] 실패한 파일 수: {len(failed_files)}")
    if failed_files:
        print("[DB] 실패한 파일 일부:", failed_files[:20])

    cur.close()
    print("[DB] 데이터 삽입 완료\n")


# ==================== 영상 검색 함수 ====================
def retrieve_video_segments(question: str, top_k: int = 3):
    """영상 세그먼트 벡터 검색"""
    print(f"\n🎬 [Video] 영상 검색 중: {question}")

    q_emb = embedding_model.embed_query(question)
    print("🧠 질문 임베딩 생성 완료")

    with get_conn() as conn:
        with conn.cursor() as cur:
            # video_segments 테이블이 있는지 확인
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'video_segments'
                );
            """)
            
            if not cur.fetchone()[0]:
                print("⚠️ video_segments 테이블이 없습니다.")
                return []
            
            cur.execute("""
                SELECT
                    video_id,
                    title,
                    start_time,
                    end_time,
                    url,
                    content
                FROM video_segments
                ORDER BY content_embedding <#> %s::vector
                LIMIT %s
            """, (q_emb, top_k))

            rows = cur.fetchall()

    print(f"📄 영상 검색 결과 수: {len(rows)}")

    results = []
    for r in rows:
        video_id, title, start, end, url, content = r
        snippet = content[:300].replace("\n", " ")

        print(f"""
   - 🎬 {title}
     ⏱ {int(start)}s ~ {int(end)}s
     {snippet}...
        """)

        results.append({
            "video_id": video_id,
            "title": title,
            "start": start,
            "end": end,
            "url": url,
            "content": content
        })

    return results


# ==================== State 정의 ====================
class GraphState(TypedDict):
    question: str
    rewritten_question: str
    route: str
    documents: List[dict]
    judgement: str
    iteration: int


# ==================== LangGraph 노드 함수들 ====================
def route_question(state: GraphState) -> GraphState:
    print("🤖 [Router] 질문 분석...\n")
    
    question = state["question"]
    keywords = [
        "창조", "성경", "하나님", "진화", "복음", "아담", "노아", 
        "대홍수", "창세기", "기독교", "세계관", "믿음", "예수님", "구원",
        "Creation", "Bible", "God", "Evolution", "Gospel", "Adam", 
        "Noah", "Great Flood", "Genesis", "Christianity", "Worldview", 
        "Faith", "Jesus", "Salvation"
    ]
    
    route = "internal" if any(k in question for k in keywords) else "internal"
    return {**state, "route": route, "iteration": 0}

def retrieve_documents(state: GraphState) -> GraphState:
    print("🌐 [Web DB] 벡터 검색 시작...\n")
    
    query = state.get("rewritten_question") or state["question"]
    q_embedding = embedding_model.embed_query(query)
    
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT title, url, content
                FROM crawled_data
                ORDER BY content_embedding <#> %s::vector
                LIMIT 5
            """, (q_embedding,))
            
            rows = cur.fetchall()
    
    docs = [{"title": r[0], "url": r[1], "content": r[2]} for r in rows]
    print(f"[Web DB] 검색 결과: {len(docs)}개\n")
    
    for i, d in enumerate(docs, start=1):
        print(f"[Web Doc {i}] {d['title'][:50]}...")
    
    print()
    return {**state, "documents": docs}

def judge_documents(state: GraphState) -> GraphState:
    print("🤖 [Judge] 문서 평가 중...\n")
    
    question = state["question"]
    docs = state["documents"]
    
    if not docs:
        return {**state, "judgement": "not_resolved"}
    
    joined_docs = "\n".join(d["content"][:500] for d in docs)
    
    prompt = f"""
    사용자 질문에 대해 아래 문서들이 충분한 정보를 제공하는지 판단하세요.

    Question: {question}
    Documents: {joined_docs}

    JSON 형식으로만 응답:
    {{"judgement": "resolved" or "not_resolved", "binary_score": "yes" or "no"}}
    """
    
    res = model.chat.completions.create(
        model="solar-pro2",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    content = res.choices[0].message.content
    
    try:
        json_start = content.find("{")
        json_obj = json.loads(content[json_start:])
        judgement = json_obj.get("judgement", "not_resolved")
    except:
        judgement = "not_resolved"
    
    print(f"[Judge] 결과: {judgement}\n")
    return {**state, "judgement": judgement}

def rewrite_question(state: GraphState) -> GraphState:
    print("✍️ [Rewrite] 질문 재작성...\n")
    
    question = state["question"]
    iteration = state.get("iteration", 0)
    
    system_rewriter = "당신은 RAG 검색 성능을 높이기 위해 질문을 더 명확하고 구체적으로 재작성하는 전문가입니다."
    
    prompt_rewriter = ChatPromptTemplate.from_messages([
        ("system", system_rewriter),
        ("human", f"Original question: {question}")
    ])
    
    chain = (
        prompt_rewriter
        | RunnableLambda(
            lambda p: model.chat.completions.create(
                model="solar-pro2",
                messages=[{"role": "user", "content": p.to_string()}],
                temperature=0
            ).choices[0].message.content
        )
        | StrOutputParser()
    )
    
    rewritten = chain.invoke({"question": question})
    print(f"[Rewrite] 재작성: {rewritten}\n")
    
    return {**state, "rewritten_question": rewritten, "iteration": iteration + 1}


# ==================== 조건부 엣지 ====================
def decide_to_rewrite(state: GraphState) -> Literal["rewrite", "end"]:
    judgement = state.get("judgement", "resolved")
    iteration = state.get("iteration", 0)
    
    if judgement == "not_resolved" and iteration < 2:
        print("✍️ [Decision] → 재작성\n")
        return "rewrite"
    else:
        print("✅ [Decision] → 검색 완료\n")
        return "end"


# ==================== LangGraph 그래프 구성 ====================
def create_graph():
    """LangGraph 그래프 생성 - 검색까지만"""
    
    workflow = StateGraph(GraphState)
    
    workflow.add_node("route", route_question)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("judge", judge_documents)
    workflow.add_node("rewrite", rewrite_question)
    
    workflow.set_entry_point("route")
    workflow.add_edge("route", "retrieve")
    workflow.add_edge("retrieve", "judge")
    
    workflow.add_conditional_edges(
        "judge",
        decide_to_rewrite,
        {
            "rewrite": "rewrite",
            "end": END
        }
    )
    
    workflow.add_edge("rewrite", "retrieve")
    
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# ============================= 언어 감지 =============================
def detect_language(text: str):
    return "ko" if any("\uac00" <= c <= "\ud7a3" for c in text) else "en"


# ==================== 통합 답변 생성 ====================
def generate(question: str) -> tuple[str, dict]:
    """웹 + 책 + 영상 통합 검색 후 답변 생성 (출처 정보 반환)"""
    print("\n" + "="*60)
    print("===== 통합 검색 시작 =====")
    print("="*60)
    print(f"💁‍♂️ 질문: {question}\n")
    
    # 1. 웹 DB 검색 (LangGraph)
    graph = create_graph()
    
    initial_state = {
        "question": question,
        "rewritten_question": "",
        "route": "",
        "documents": [],
        "judgement": "",
        "iteration": 0
    }
    
    config = {"configurable": {"thread_id": "1"}}
    web_result = graph.invoke(initial_state, config)
    web_docs = web_result["documents"]
    
    # 2. 책 DB 검색
    book_docs = retrieve_pages(question, top_k=3)
    
    # 3. 영상 DB 검색
    video_docs = retrieve_video_segments(question, top_k=3)
    
    # 4. 문서 없으면 종료
    if not web_docs and not book_docs and not video_docs:
        return "📘 관련 정보를 찾을 수 없습니다.", {}
    
    # 5. 언어 감지
    lang = detect_language(question)
    lang_instruction = "한국어로 답변하세요." if lang == "ko" else "Answer in English."
    
    # 6. 컨텍스트 구성
    context_parts = []
    
    if video_docs:
        context_parts.append("=" * 50)
        context_parts.append("🎬 영상 자료")
        context_parts.append("=" * 50)
        for i, doc in enumerate(video_docs, 1):
            context_parts.append(f"\n[영상 {i}] {doc['title']}")
            context_parts.append(f"시간: {int(doc['start'])}초 ~ {int(doc['end'])}초")
            context_parts.append(f"내용: {doc['content'][:800]}")
    
    if web_docs:
        context_parts.append("\n" + "=" * 50)
        context_parts.append("📰 웹사이트 자료")
        context_parts.append("=" * 50)
        for i, doc in enumerate(web_docs, 1):
            context_parts.append(f"\n[웹 문서 {i}]")
            context_parts.append(f"제목: {doc['title']}")
            context_parts.append(f"내용: {doc['content'][:800]}")
    
    if book_docs:
        context_parts.append("\n" + "=" * 50)
        context_parts.append("📖 책 자료")
        context_parts.append("=" * 50)
        for i, doc in enumerate(book_docs, 1):
            context_parts.append(f"\n[{doc['book']} - 페이지 {doc['page']}]")
            context_parts.append(f"내용: {doc['content'][:800]}")
    
    context = "\n".join(context_parts)
    
    # 7. 답변 생성
    system_prompt = f"""
    당신은 기독교적 세계관과 창조과학에 기반한 전문가입니다.

    규칙:
    - 반드시 제공된 자료[영상 자료, 웹 자료, 책 자료]만 모두 활용
    - 🌍 과학적 관점 / 📜 성경적 관점으로 구분
    - 명확하고 이해하기 쉽게 작성

    {lang_instruction}
    """

    user_prompt = f"""
    [자료]
    {context}

    [질문]
    {question}
    """

    print("🤖 [Generate] 통합 답변 생성 중...\n")
    res = model.chat.completions.create(
        model="solar-pro2",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    
    answer = res.choices[0].message.content
    
    # 8. 출처 정보 구성
    sources_info = {
        "video_docs": video_docs,
        "web_docs": web_docs,
        "book_docs": book_docs
    }
    
    print("✅ 통합 답변 완료!\n")
    print("="*60 + "\n")
    return answer, sources_info


# ==================== 테스트 ====================
if __name__ == "__main__":
    test_question = "창조과학이 뭔가요?"
    answer, sources = generate(test_question)
    print(f"\n최종 답변:\n{answer}")