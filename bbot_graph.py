import json
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
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
    question: str
    rewritten_question: str
    route: str
    documents: List[dict]
    judgement: str
    iteration: int


# ==================== Utility ====================

def format_timedelta(seconds: int) -> str:
    td = timedelta(seconds=int(seconds))
    total = int(td.total_seconds())
    h, r = divmod(total, 3600)
    m, s = divmod(r, 60)
    return f"{h:02}:{m:02}:{s:02}"


def detect_language(text: str) -> str:
    return "ko" if any("\uac00" <= c <= "\ud7a3" for c in text) else "en"


# ==================== 병렬 통합 검색 ====================

def retrieve_all_documents_parallel(question: str, top_k: int = 3):
    print("🔍 [Retrieve] Parallel search..\n")

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_web = executor.submit(
            retrieve_web_documents,
            question,
            top_k
        )

        future_book = executor.submit(
            retrieve_pages,
            question,
            top_k
        )

        future_video = executor.submit(
            retrieve_video_segments,
            question,
            top_k
        )

        web_docs = future_web.result()
        book_docs = future_book.result()
        video_docs = future_video.result()

    print("✅ Parallel search completed\n")

    return {
        "web_docs": web_docs or [],
        "book_docs": book_docs or [],
        "video_docs": video_docs or [],
        "all_docs": (web_docs or []) + (book_docs or []) + (video_docs or [])
    }


# ==================== Graph Nodes ====================

def route_question(state: GraphState) -> GraphState:
    print("🤖 [Router] Question routing...\n")

    return {
        **state,
        "route": "internal",
        "iteration": 0
    }


def retrieve_documents(state: GraphState) -> GraphState:
    print("🌐 [Retrieve] Parallel search...\n")

    query = state.get("rewritten_question") or state["question"]

    result = retrieve_all_documents_parallel(
        query,
        top_k=3
    )

    return {
        **state,
        "documents": result["all_docs"]
    }


def judge_documents(state: GraphState) -> GraphState:
    print("🤖 [Judge] Evaluating documents...\n")

    docs = state.get("documents", [])

    if not docs:
        print("[Judge] No documents found → not_resolved\n")

        return {
            **state,
            "judgement": "not_resolved"
        }

    joined_docs = "\n".join(
        doc.get("content", "")[:500]
        for doc in docs
        if doc.get("content")
    )

    prompt = f"""
사용자 질문에 대해 아래 문서들이 충분한 정보를 제공하는지 판단하세요.

Question:
{state["question"]}

Documents:
{joined_docs}

반드시 JSON 형식으로만 응답하세요.

예시:
{{"judgement": "resolved"}}

또는

{{"judgement": "not_resolved"}}
"""

    res = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    try:
        content = res.choices[0].message.content
        json_obj = json.loads(
            content[content.find("{"):]
        )
        judgement = json_obj.get(
            "judgement",
            "not_resolved"
        )

    except Exception:
        judgement = "not_resolved"

    print(f"[Judge] Result: {judgement}\n")

    return {
        **state,
        "judgement": judgement
    }


def rewrite_question(state: GraphState) -> GraphState:
    print("✍️ [Rewrite] Question rewriting...\n")

    question = state["question"]
    iteration = state.get("iteration", 0)

    prompt_rewriter = ChatPromptTemplate.from_messages([
        (
            "system",
            "당신은 RAG 검색 성능을 높이기 위해 질문을 더 명확하고 구체적으로 재작성하는 전문가입니다."
        ),
        (
            "human",
            f"Original question: {question}"
        )
    ])

    chain = (
        prompt_rewriter
        | RunnableLambda(
            lambda p: client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": p.to_string()
                    }
                ],
                temperature=0
            ).choices[0].message.content
        )
        | StrOutputParser()
    )

    rewritten = chain.invoke({
        "question": question
    })

    print(f"[Rewrite] Rewritten question: {rewritten}\n")

    return {
        **state,
        "rewritten_question": rewritten,
        "iteration": iteration + 1
    }


# ==================== Conditional Edge ====================

def decide_to_rewrite(
    state: GraphState
) -> Literal["rewrite", "end"]:

    if (
        state.get("judgement") == "not_resolved"
        and state.get("iteration", 0) < 2
    ):
        print("✍️ [Decision] → Rewrite\n")
        return "rewrite"

    print("✅ [Decision] → Search completed\n")
    return "end"


# ==================== Graph Build ====================

def create_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node(
        "route",
        route_question
    )

    workflow.add_node(
        "retrieve",
        retrieve_documents
    )

    workflow.add_node(
        "judge",
        judge_documents
    )

    workflow.add_node(
        "rewrite",
        rewrite_question
    )

    workflow.set_entry_point("route")

    workflow.add_edge(
        "route",
        "retrieve"
    )

    workflow.add_edge(
        "retrieve",
        "judge"
    )

    workflow.add_conditional_edges(
        "judge",
        decide_to_rewrite,
        {
            "rewrite": "rewrite",
            "end": END
        }
    )

    workflow.add_edge(
        "rewrite",
        "retrieve"
    )

    return workflow.compile(
        checkpointer=MemorySaver()
    )


# ==================== Final Generate ====================

def generate(question: str):
    print("\n" + "=" * 60)
    print("===== Integrated Search Started =====")
    print("=" * 60)
    print(f"💁‍♂️ Question: {question}\n")

    graph = create_graph()

    graph_result = graph.invoke(
        {
            "question": question,
            "rewritten_question": "",
            "route": "",
            "documents": [],
            "judgement": "",
            "iteration": 0
        },
        {
            "configurable": {
                "thread_id": "1"
            }
        }
    )

    all_docs = graph_result.get("documents", [])

    if not all_docs:
        return "📘 관련 정보를 찾을 수 없습니다.", {}

    web_docs = []
    book_docs = []
    video_docs = []

    for doc in all_docs:
        if "url" in doc:
            web_docs.append(doc)
        elif "book" in doc:
            book_docs.append(doc)
        elif "start" in doc and "end" in doc:
            video_docs.append(doc)

    lang_instruction = (
        "한국어로 답변하세요."
        if detect_language(question) == "ko"
        else "Answer in English."
    )

    context_parts = []

    if video_docs:
        context_parts.append("🎬 Video Resources")

        for i, doc in enumerate(video_docs, 1):
            context_parts.append(
                f"[Video {i}] "
                f"{doc.get('title', '')} "
                f"({format_timedelta(doc.get('start', 0))}"
                f"~{format_timedelta(doc.get('end', 0))})"
            )
            context_parts.append(
                doc.get("content", "")[:800]
            )

    if web_docs:
        context_parts.append("📰 Web Resources")

        for i, doc in enumerate(web_docs, 1):
            context_parts.append(
                f"[Web {i}] {doc.get('title', '')}"
            )
            context_parts.append(
                doc.get("content", "")[:800]
            )

    if book_docs:
        context_parts.append("📖 Book Resources")

        for i, doc in enumerate(book_docs, 1):
            context_parts.append(
                f"[{doc.get('book', '')} "
                f"p{doc.get('page', '')}]"
            )
            context_parts.append(
                doc.get("content", "")[:800]
            )

    context = "\n".join(context_parts)

    system_prompt = f"""
당신은 기독교적 세계관과 창조과학에 기반한 전문가입니다.

규칙:
- 반드시 제공된 자료만 활용
- 🌍 과학적 관점 / 📜 성경적 관점으로 구분
- 서론 없이 바로 답변
- 명확하고 이해하기 쉽게 작성

{lang_instruction}
"""

    print("🤖 [Generate] 답변 생성 중...\n")

    res = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content":
                    f"[자료]\n{context}\n\n"
                    f"[질문]\n{question}"
            }
        ],
        temperature=0
    )

    answer = res.choices[0].message.content

    print("✅ Integrated answer completed!\n")

    return answer, {
        "video_docs": video_docs,
        "web_docs": web_docs,
        "book_docs": book_docs
    }