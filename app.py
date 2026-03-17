import streamlit as st
import json
from datetime import datetime, timedelta

from bbotCss import CSS
from bbot_graph import generate
from config import get_conn
from db_init import init_all

st.markdown(CSS, unsafe_allow_html=True)

# ==================== 페이지 설정 ====================
st.set_page_config(
    page_title="BeBot", page_icon="💭", layout="wide", initial_sidebar_state="expanded"
)


# ==================== DB 체크 ====================
def table_exists(table_name: str) -> bool:
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT 1
                        FROM information_schema.tables
                        WHERE table_schema = 'public'
                          AND table_name = %s
                    );
                """, (table_name,))
                return cur.fetchone()[0]
    except Exception as e:
        print(f"❌ 테이블 존재 확인 에러 ({table_name}): {e}")
        return False


def get_db_stats():
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM crawled_data;")
                total_docs = cur.fetchone()[0]
                cur.execute("SELECT COUNT(DISTINCT title) FROM crawled_data;")
                unique_titles = cur.fetchone()[0]
                return total_docs, unique_titles
    except Exception as e:
        print(f"❌ DB 통계 조회 에러: {e}")
        return 0, 0


# ==================== 세션 상태 초기화 ====================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "db_ready" not in st.session_state:
    st.session_state.db_ready = False
if "show_workflow" not in st.session_state:
    st.session_state.show_workflow = False


# ==================== utils ====================
def format_timedelta(td: timedelta) -> str:
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


# ==================== 사이드바 ====================
with st.sidebar:
    st.markdown(
        '<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQtl4elGepo51WtnezrB1eiyE0TS2GGejnufA&s" '
        'style="display:block; margin:auto; width:150px; border-radius:10px;">',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<h3 style='text-align: center;'>🤖 BeBot - Let there be light 🤖</h3>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ==================== DB 자동 초기화 ====================
    if not st.session_state.db_ready:
        with st.spinner("🔍 DB 확인 중..."):
            all_exist = (
                table_exists("crawled_data") and
                table_exists("book_eng") and
                table_exists("video_db")
            )

        if not all_exist:
            st.info("⚙️ DB 초기화 중... (최초 1회, 수 분 소요)")
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("DB 생성 중...")
            progress_bar.progress(50)

            try:
                init_all()  # ✅ db_init.py의 통합 초기화 함수 호출
                st.session_state.db_ready = True
                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()
                st.success("✅ DB 준비 완료!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ DB 생성 실패: {e}")
                st.stop()
        else:
            st.session_state.db_ready = True
            st.rerun()

    else:
        st.success("✅ DB 사용 가능")

        try:
            total_docs, unique_titles = get_db_stats()
            st.markdown("### 📊 DB 통계")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("총 문서", f"{total_docs:,}")
            with col2:
                st.metric("고유 자료", f"{unique_titles:,}")
        except Exception as e:
            st.warning(f"통계 로드 실패: {e}")

        st.markdown("---")

    # 설정 옵션
    st.markdown("### ⚙️ 설정")
    st.session_state.show_workflow = st.checkbox(
        "워크플로우 표시",
        value=st.session_state.show_workflow,
        help="LangGraph 실행 과정을 실시간으로 표시합니다",
    )
    temperature = st.slider(
        "창의성 수준", min_value=0.0, max_value=1.0, value=0.0, step=0.1,
        help="높을수록 더 창의적이지만 덜 정확할 수 있습니다",
    )
    st.markdown("---")

    # 대화 관리
    st.markdown("### 💬 대화 관리")
    if st.button("🗑️ 대화 초기화", type="secondary", key="clear_chat"):
        st.session_state.messages = []
        st.rerun()

    if st.button("💾 대화 저장"):
        if st.session_state.messages:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_history_{timestamp}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(st.session_state.messages, f, ensure_ascii=False, indent=2)
            st.success(f"✅ {filename}에 저장됨!")
        else:
            st.warning("저장할 대화가 없습니다.")

    st.markdown("---")
    if st.button("🔍 워크플로우 구조 보기"):
        with st.expander("LangGraph 구조", expanded=True):
            st.code("""
            route → retrieve → judge
                              ↓
                           [resolved?]
                        ↙          ↘
                   generate      rewrite
                      ↓              ↓
                     END        retrieve
                                    ↓
                                  judge
            """, language="text")

    st.markdown("---")
    st.caption("Powered by LangGraph 🦜🔗")


# ==================== 메인 영역 ====================
st.markdown('<div class="main-title">BeBot Q&A</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Ask questions about the biblical worldview and creation science</div>',
    unsafe_allow_html=True,
)

# 예시 질문 버튼
if not st.session_state.messages:
    st.markdown("### 💡 Example Questions")
    col1, col2, col3 = st.columns(3)
    example_prompt = None
    with col1:
        if st.button("🌍 창조과학이란?", use_container_width=True):
            example_prompt = "창조과학이 무엇인가요?"
    with col2:
        if st.button("🦴 화석은 어떻게?", use_container_width=True):
            example_prompt = "화석은 진화론을 지지하나요?"
    with col3:
        if st.button("📖 창세기 해석", use_container_width=True):
            example_prompt = "창세기 1장을 어떻게 이해해야 하나요?"
    st.markdown("---")
else:
    example_prompt = None


# ==================== 채팅 인터페이스 ====================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and "video_source" in msg and msg["video_source"]:
            video_info = msg["video_source"]
            st.markdown("---")
            st.markdown("### 🎬 참고 영상")
            if "youtu.be" in video_info["url"] or "youtube.com" in video_info["url"]:
                video_id = (
                    video_info["url"].split("youtu.be/")[-1].split("?")[0]
                    if "youtu.be" in video_info["url"]
                    else video_info["url"].split("v=")[-1].split("&")[0]
                )
                embed_url = f"https://www.youtube.com/embed/{video_id}?start={int(video_info['start'])}"
                st.markdown(f"**{video_info['title']}**")
                st.markdown(
                    f'<iframe width="100%" height="400" src="{embed_url}" frameborder="0" allowfullscreen></iframe>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f"**{video_info['title']}**")
                st.markdown(f"[영상 보기]({video_info['url']}) (시작: {int(video_info['start'])}초)")

        if "workflow" in msg and st.session_state.show_workflow:
            with st.expander("🔍 처리 과정 보기"):
                for step in msg["workflow"]:
                    st.markdown(f'<div class="workflow-step">✓ {step}</div>', unsafe_allow_html=True)


# ==================== 사용자 입력 처리 ====================
prompt = st.chat_input("Curious about creation science ✨") or example_prompt

if prompt:
    if not st.session_state.db_ready:
        st.error("⚠️ DB가 준비되지 않았습니다.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        workflow_steps = []
        if st.session_state.show_workflow:
            workflow_container = st.empty()

        with st.spinner("Searching..."):
            try:
                response, sources_info = generate(prompt)
                workflow_steps = [
                    "질문 라우팅 완료",
                    "벡터 검색 수행 (웹 + 책 + 영상)",
                    "문서 적합성 판단",
                    "답변 생성 완료",
                ]

                if st.session_state.show_workflow:
                    with workflow_container:
                        with st.expander("🔍 Processing", expanded=True):
                            for step in workflow_steps:
                                st.markdown(f'<div class="workflow-step">✓ {step}</div>', unsafe_allow_html=True)

                st.markdown(response)

                # 영상 임베드
                video_source = None
                if sources_info.get("video_docs"):
                    top_video = sources_info["video_docs"][0]
                    video_source = {"title": top_video["title"], "url": top_video["url"], "start": top_video["start"]}
                    st.markdown("---")
                    st.markdown("### 🎬 참고 영상")
                    if "youtu.be" in top_video["url"] or "youtube.com" in top_video["url"]:
                        video_id = (
                            top_video["url"].split("youtu.be/")[-1].split("?")[0]
                            if "youtu.be" in top_video["url"]
                            else top_video["url"].split("v=")[-1].split("&")[0]
                        )
                        embed_url = f"https://www.youtube.com/embed/{video_id}?start={int(top_video['start'])}"
                        st.markdown(f"**{top_video['title']}**")
                        st.markdown(
                            f'<iframe width="100%" height="400" src="{embed_url}" frameborder="0" allowfullscreen></iframe>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(f"**{top_video['title']}**")
                        st.markdown(f"[영상 보기]({top_video['url']}) (시작: {int(top_video['start'])}초)")

                # 출처 정보
                sources = []
                if sources_info.get("web_docs"):
                    web_urls = list(set([d["url"] for d in sources_info["web_docs"] if d.get("url")]))
                    if web_urls:
                        sources.append("\n**🌐 웹 자료:**")
                        for url in web_urls:
                            sources.append(f"- {url}")

                if sources_info.get("book_docs"):
                    book_names = list(set([d["book"] for d in sources_info["book_docs"]]))
                    sources.append("\n**📖 책 자료:**")
                    if len(book_names) == 1:
                        pages = ", ".join("p" + str(d["page"]) for d in sources_info["book_docs"])
                        sources.append(f"- {book_names[0]} ({pages})")
                    else:
                        for doc in sources_info["book_docs"]:
                            sources.append(f"- {doc['book']} (p{doc['page']})")

                if sources_info.get("video_docs"):
                    sources.append("\n**🎬 영상 자료:**")
                    for doc in sources_info["video_docs"]:
                        start = format_timedelta(timedelta(seconds=int(doc["start"])))
                        end = format_timedelta(timedelta(seconds=int(doc["end"])))
                        sources.append(f"- [{doc['title']}]({doc['url']}) ({start}-{end})")

                if sources:
                    st.markdown("\n---\n" + "\n".join(sources))

            except Exception as e:
                st.error(f"❌ 오류 발생: {str(e)}")
                response = "죄송합니다. 답변 생성 중 오류가 발생했습니다."
                workflow_steps = []
                video_source = None

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "workflow": workflow_steps,
        "video_source": video_source,
    })


# ==================== 푸터 ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>🙏 All answers are based on a biblical perspective</p>
    <p>📚 Source: Korean Creation Science Association & related sources</p>
</div>
""", unsafe_allow_html=True)



