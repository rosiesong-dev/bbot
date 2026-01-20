# bbot - web 기반 RAG 
import os
import json
from datetime import datetime
from typing import List, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel
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

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    port=os.getenv("DB_PORT")
)


# ==================== DB 관련 함수 ====================
enc = tiktoken.get_encoding("cl100k_base")     # tokenizer

def count_tokens(text: str) -> int:            # 토큰 수 세기
    return len(enc.encode(text))               # 토큰 수 반환

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

    # PostgreSQL 연결
    cur = conn.cursor()
    print("[DB] 연결 성공")

    # pgvector 확장 
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # 테이블 생성
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
            # 파일 내용 읽기
            with open(os.path.join(folder_path, fname), "r", encoding="utf-8") as f:
                full_text = f.read()

            # 파일 파싱
            lines = full_text.split('\n')
            
            title = ""
            url = ""
            crawl_time = None
            content = ""
            
            # 각 줄 파싱
            content_start_idx = 0
            for i, line in enumerate(lines):
                if line.startswith("Title:"):
                    title = line.replace("Title:", "").strip()
                elif line.startswith("URL:"):
                    url = line.replace("URL:", "").strip()
                elif line.startswith("Crawl Time:"):
                    crawl_time_str = line.replace("Crawl Time:", "").strip()
                    try:
                        # ISO 8601 형식 파싱
                        crawl_time = datetime.fromisoformat(crawl_time_str.replace('+09:00', ''))
                    except:
                        crawl_time = datetime.now()
                elif line.startswith("Content:"):
                    # Content: 다음 줄부터가 실제 본문
                    content_start_idx = i + 1
                    break
            
            # 본문 추출 (Content: 이후 모든 내용)
            if content_start_idx > 0:
                content = '\n'.join(lines[content_start_idx:]).strip()
            else:
                # Content: 태그가 없으면 전체를 본문으로
                content = full_text
            
            # title이 비어있으면 파일명 사용
            if not title:
                title = fname.replace(".txt", "")
            
            # 본문을 청크로 분할
            chunks = split_text_by_tokens(content, max_tokens=max_tokens)

            # 각 청크를 DB에 삽입
            for chunk_idx, chunk in enumerate(chunks):
                # 임베딩 생성
                embedding_vector = embedding_model.embed_query(chunk)

                cur.execute(
                    "INSERT INTO crawled_data (title, url, crawl_time, content, content_embedding) VALUES (%s, %s, %s, %s, %s::vector)",
                    (title, url, crawl_time, chunk, embedding_vector)
                )
                inserted_count += 1

            # 진행 상황 출력 (더 상세하게)
            if idx % 10 == 0:
                print(f"[DB] {idx}개 파일 처리 완료")
                print(f"   최근 파일: {title[:50]}...")
                print(f"   URL: {url[:80] if url else '(없음)'}")

        except Exception as e:
            print(f"[ERROR] 파일 삽입 실패: {fname}")
            print(f"   에러 내용: {e}")
            failed_files.append(fname)
            conn.rollback()

    # 변경사항 커밋
    conn.commit()

    # 최종 통계 출력
    cur.execute("SELECT COUNT(*) FROM crawled_data;")
    total_count = cur.fetchone()[0]
    print(f"\n[DB] ===== 최종 통계 =====")
    print(f"[DB] 총 데이터 개수: {total_count}")
    print(f"[DB] 성공적으로 삽입된 청크 수: {inserted_count}")
    print(f"[DB] 실패한 파일 수: {len(failed_files)}")
    if failed_files:
        print("[DB] 실패한 파일 일부:", failed_files[:20])

    # 연결 종료
    cur.close()
    print("[DB] 데이터 삽입 완료\n")


# ==================== State 정의 ====================
class GraphState(TypedDict):
    """LangGraph의 상태를 정의하는 클래스"""
    question: str                    # 원본 질문
    rewritten_question: str          # 재작성된 질문
    route: str                       # 라우팅 결과
    documents: List[dict]            # 검색된 문서들
    judgement: str                   # 검색 결과 판단
    answer: str                      # 최종 답변
    iteration: int                   # 재시도 횟수



# ==================== LangGraph 노드 함수들 ====================

def route_question(state: GraphState) -> GraphState:
    """라우팅 노드: 질문이 창조과학/성경 관련인지 판단"""
    print("🤖 [Node: Router] 질문 의도 분석 중...")
    
    question = state["question"]
    keywords = [
        "창조", "성경", "하나님", "진화", "복음", "아담", "노아", 
        "대홍수", "창세기", "기독교", "세계관", "믿음", "예수님", "구원",
        "Creation", "Bible", "God", "Evolution", "Gospel", "Adam", 
        "Noah", "Great Flood", "Genesis", "Christianity", "Worldview", 
        "Faith", "Jesus", "Salvation"
    ]
    
    route = "internal" if any(k in question for k in keywords) else "internal"
    print(f"[Router] 선택된 경로: {route}")
    
    return {**state, "route": route, "iteration": 0}


def retrieve_documents(state: GraphState) -> GraphState:
    """문서 검색 노드"""
    print("🤖 [Node: Retrieve] 벡터 검색 시작")
    
    # 재작성된 질문이 있으면 사용, 없으면 원본 사용
    query = state.get("rewritten_question") or state["question"]
    
    q_embedding = embedding_model.embed_query(query)
    
    # PostgreSQL에서 벡터 유사도 검색
    cur = conn.cursor()
    
    cur.execute("""
        SELECT title, url, content
        FROM crawled_data
        ORDER BY content_embedding <#> %s::vector
        LIMIT 5
    """, (q_embedding,))
    
    rows = cur.fetchall()
    cur.close()
    
    docs = [{"title": r[0], "url": r[1], "content": r[2]} for r in rows]
    print(f"[Retrieve] 검색 문서 수: {len(docs)}")
    
    
    for i, d in enumerate(docs, start=1):
        print(f"\n[Doc {i}]")
        print(f"Title: {d['title']}")
        print(f"URL: {d['url']}")
        print(f"Content (preview): {d['content'][:300]}")  # 앞 300자만

    return {**state, "documents": docs}


def judge_documents(state: GraphState) -> GraphState:
    """문서 판단 노드"""
    print("🤖 [Node: Judge] 검색 결과 평가 중...")
    
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
    {{
        "judgement": "resolved" or "not_resolved",
        "binary_score": "yes" or "no"
    }}
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
    
    print(f"[Judge] 판단 결과: {judgement}")
    return {**state, "judgement": judgement}


def rewrite_question(state: GraphState) -> GraphState:
    """질문 재작성 노드"""
    print("🤖 [Node: Rewrite] 질문 재작성 중...")
    
    question = state["question"]
    iteration = state.get("iteration", 0)
    
    system_rewriter = """
    당신은 RAG 검색 성능을 높이기 위해 질문을 더 명확하고 구체적으로 재작성하는 전문가입니다.
    """
    
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
    print(f"[Rewrite] 재작성된 질문: {rewritten}")
    
    return {**state, "rewritten_question": rewritten, "iteration": iteration + 1}



def generate_answer(state: GraphState) -> GraphState:
    """답변 생성 노드 - DB에서 가져온 URL을 자동으로 추가"""
    print("🤖 [Node: Generate] 답변 생성 중...")
    
    question = state["question"]
    docs = state["documents"]
    
    if not docs:
        return {**state, "answer": "제공된 문서에는 해당 정보가 없습니다."}
    
    # 언어 감지
    lang = "ko" if any('\uac00' <= ch <= '\ud7a3' for ch in question) else "en"
    lang_instruction = "한국어로 답변하세요." if lang == "ko" else "Answer in English."
    
    # 컨텍스트 구성 (URL은 LLM에게 보내지 않음)
    context = "\n\n".join(
        f"[문서 {i+1}]\n제목: {d['title']}\n내용: {d['content']}"
        for i, d in enumerate(docs)
    )
    
    system_prompt = f"""
    당신은 기독교적 세계관과 창조론에 기반해 답변하는 전문가입니다.

    규칙:
    - 반드시 제공된 [문서] 내용만 사용하세요.
    - 🌍 과학적 관점과 📜 성경적 관점으로 구분하여 설명하세요.


    {lang_instruction}
    """
    
    user_prompt = f"""
    [문서]
    {context}

    [질문]
    {question}
    """
    
    res = model.chat.completions.create(
        model="solar-pro2",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0
    )
    
    answer = res.choices[0].message.content
    
    # DB에서 가져온 실제 URL 자동 추가
    unique_urls = list(set([d["url"] for d in docs if d["url"]]))  # 중복 제거
    
    if unique_urls:
        url_section = "\n\n---\n**📚 참고 웹사이트 링크:**\n"
        for i, url in enumerate(unique_urls, 1):
            url_section += f"{i}. {url}\n"
        
        answer = answer + url_section
    
    print("✅ 답변 생성 완료 (URL 자동 추가됨)")
    
    return {**state, "answer": answer}



# ==================== 조건부 엣지 ====================
def decide_to_rewrite(state: GraphState) -> Literal["rewrite", "generate"]:
    """재작성 여부 결정"""
    judgement = state.get("judgement", "resolved")
    iteration = state.get("iteration", 0)
    
    # 최대 2번까지만 재시도
    if judgement == "not_resolved" and iteration < 2:
        print("✍️ [Decision] → 질문 재작성")
        return "rewrite"
    else:
        print("✍️ [Decision] → 답변 생성")
        return "generate"




# ==================== LangGraph 그래프 구성 ====================
def create_graph():
    """LangGraph 그래프 생성"""
    
    workflow = StateGraph(GraphState)
    
    # 노드 추가
    workflow.add_node("route", route_question)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("judge", judge_documents)
    workflow.add_node("rewrite", rewrite_question)
    workflow.add_node("generate", generate_answer)
    
    # 엣지 연결
    workflow.set_entry_point("route")
    workflow.add_edge("route", "retrieve")
    workflow.add_edge("retrieve", "judge")
    
    # 조건부 엣지
    workflow.add_conditional_edges(
        "judge",
        decide_to_rewrite,
        {
            "rewrite": "rewrite",
            "generate": "generate"
        }
    )
    
    # 재작성 후 다시 검색
    workflow.add_edge("rewrite", "retrieve")
    
    # 생성 후 종료
    workflow.add_edge("generate", END)
    
    # 메모리 추가 (선택사항)
    memory = MemorySaver()
    
    return workflow.compile(checkpointer=memory)


# ==================== Main 함수 ====================

def generate(question: str) -> str:
    """메인 함수: 질문을 받아 답변 생성"""
    print("\n===== NEW QUERY (LangGraph) =====")
    print(f"💁‍♂️ User Question: {question}")
    
    # 그래프 생성
    graph = create_graph()
    
    # 초기 상태
    initial_state = {
        "question": question,
        "rewritten_question": "",
        "route": "",
        "documents": [],
        "judgement": "",
        "answer": "",
        "iteration": 0
    }
    
    # 그래프 실행
    config = {"configurable": {"thread_id": "1"}}
    final_state = graph.invoke(initial_state, config)
    
    print("[Done] 응답 완료\n")
    return final_state["answer"]


# ==================== 그래프 시각화 ====================

def visualize_graph():
    """그래프 구조를 Mermaid 다이어그램으로 출력"""
    graph = create_graph()
    try:
        print(graph.get_graph().draw_mermaid())
    except Exception as e:
        print(f"시각화 실패: {e}")
        print("graphviz 또는 mermaid 라이브러리가 필요합니다.")


# ==================== 테스트 ====================
if __name__ == "__main__":
    test_question = "창조과학이 뭔가요?"
    answer = generate(test_question)
    print(f"\n최종 답변:\n{answer}")