import json
from typing import List, Literal
from typing_extensions import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from config import LLM_MODEL
from llm_factory import get_client
from bbot_web import retrieve_web_documents
from bbot_book import retrieve_pages
from bbot_video import retrieve_video_segments

client = get_client()


# ==================== State ====================
class GraphState(TypedDict):
    question:           str
    rewritten_question: str
    route:              str
    documents:          List[dict]
    judgement:          str
    iteration:          int


# ==================== 노드 ====================
def route_question(state: GraphState) -> GraphState:
    print("🤖 [Router] 질문 분석...\n")
    return {**state, "route": "internal", "iteration": 0}


def retrieve_documents(state: GraphState) -> GraphState:
    print("🌐 [Retrieve] 웹 DB 검색...\n")
    query = state.get("rewritten_question") or state["question"]
    docs  = retrieve_web_documents(query, top_k=5)
    return {**state, "documents": docs}


def judge_documents(state: GraphState) -> GraphState:
    print("🤖 [Judge] 문서 평가 중...\n")
    docs = state["documents"]
    if not docs:
        return {**state, "judgement": "not_resolved"}

    joined_docs = "\n".join(d["content"][:500] for d in docs)
    prompt = f"""
사용자 질문에 대해 아래 문서들이 충분한 정보를 제공하는지 판단하세요.
Question: {state["question"]}
Documents: {joined_docs}
JSON 형식으로만 응답:
{{"judgement": "resolved" or "not_resolved", "binary_score": "yes" or "no"}}
"""
    res = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    try:
        content  = res.choices[0].message.content
        json_obj = json.loads(content[content.find("{"):])
        judgement = json_obj.get("judgement", "not_resolved")
    except Exception:
        judgement = "not_resolved"

    print(f"[Judge] 결과: {judgement}\n")
    return {**state, "judgement": judgement}


def rewrite_question(state: GraphState) -> GraphState:
    print("✍️ [Rewrite] 질문 재작성...\n")
    question  = state["question"]
    iteration = state.get("iteration", 0)

    prompt_rewriter = ChatPromptTemplate.from_messages([
        ("system", "당신은 RAG 검색 성능을 높이기 위해 질문을 더 명확하고 구체적으로 재작성하는 전문가입니다."),
        ("human", f"Original question: {question}"),
    ])
    chain = (
        prompt_rewriter
        | RunnableLambda(lambda p: client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": p.to_string()}],
            temperature=0,
        ).choices[0].message.content)
        | StrOutputParser()
    )
    rewritten = chain.invoke({"question": question})
    print(f"[Rewrite] 재작성: {rewritten}\n")
    return {**state, "rewritten_question": rewritten, "iteration": iteration + 1}


# ==================== 조건부 엣지 ====================
def decide_to_rewrite(state: GraphState) -> Literal["rewrite", "end"]:
    if state.get("judgement") == "not_resolved" and state.get("iteration", 0) < 2:
        print("✍️ [Decision] → 재작성\n")
        return "rewrite"
    print("✅ [Decision] → 검색 완료\n")
    return "end"


# ==================== 그래프 ====================
def create_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("route",    route_question)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("judge",    judge_documents)
    workflow.add_node("rewrite",  rewrite_question)

    workflow.set_entry_point("route")
    workflow.add_edge("route",    "retrieve")
    workflow.add_edge("retrieve", "judge")
    workflow.add_conditional_edges(
        "judge", decide_to_rewrite,
        {"rewrite": "rewrite", "end": END},
    )
    workflow.add_edge("rewrite", "retrieve")
    return workflow.compile(checkpointer=MemorySaver())


# ==================== 언어 감지 ====================
def detect_language(text: str) -> str:
    return "ko" if any("\uac00" <= c <= "\ud7a3" for c in text) else "en"


# ==================== 통합 답변 생성 ====================
def generate(question: str) -> tuple[str, dict]:
    print("\n" + "="*60)
    print("===== 통합 검색 시작 =====")
    print("="*60)
    print(f"💁‍♂️ 질문: {question}\n")

    # 1. 웹 DB (LangGraph)
    graph      = create_graph()
    web_result = graph.invoke(
        {"question": question, "rewritten_question": "", "route": "",
         "documents": [], "judgement": "", "iteration": 0},
        {"configurable": {"thread_id": "1"}},
    )
    web_docs   = web_result["documents"]
    book_docs  = retrieve_pages(question, top_k=3)
    video_docs = retrieve_video_segments(question, top_k=3)

    if not web_docs and not book_docs and not video_docs:
        return "📘 관련 정보를 찾을 수 없습니다.", {}

    lang_instruction = (
        "한국어로 답변하세요." if detect_language(question) == "ko" else "Answer in English."
    )

    # 컨텍스트 구성
    context_parts = []
    if video_docs:
        context_parts.append("🎬 영상 자료")
        for i, doc in enumerate(video_docs, 1):
            context_parts.append(f"[영상 {i}] {doc['title']} ({int(doc['start'])}s~{int(doc['end'])}s)")
            context_parts.append(doc["content"][:800])
    if web_docs:
        context_parts.append("📰 웹사이트 자료")
        for i, doc in enumerate(web_docs, 1):
            context_parts.append(f"[웹 {i}] {doc['title']}")
            context_parts.append(doc["content"][:800])
    if book_docs:
        context_parts.append("📖 책 자료")
        for i, doc in enumerate(book_docs, 1):
            context_parts.append(f"[{doc['book']} p{doc['page']}]")
            context_parts.append(doc["content"][:800])

    context = "\n".join(context_parts)

    system_prompt = f"""당신은 기독교적 세계관과 창조과학에 기반한 전문가입니다.
규칙:
- 반드시 제공된 자료만 활용
- 🌍 과학적 관점 / 📜 성경적 관점으로 구분, 서론 금지
- 명확하고 이해하기 쉽게 작성
{lang_instruction}"""

    print("🤖 [Generate] 답변 생성 중...\n")
    res = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"[자료]\n{context}\n\n[질문]\n{question}"},
        ],
        temperature=0,
    )
    answer = res.choices[0].message.content
    print("✅ 통합 답변 완료!\n")
    return answer, {"video_docs": video_docs, "web_docs": web_docs, "book_docs": book_docs}