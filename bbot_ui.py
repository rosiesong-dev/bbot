import streamlit as st
import json
from bbot_web import create_db, generate
from bbot_book import create_book_db
from bbot_video import create_video_db_from_folder
import os
import psycopg2
from dotenv import load_dotenv
from datetime import datetime

from bbotCss import CSS
st.markdown(CSS, unsafe_allow_html=True)


# ==================== 환경 변수 로드 ====================
load_dotenv() 


# ==================== 페이지 설정 ====================
st.set_page_config(
    page_title="BeBot",
    page_icon="💭",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================== DB 연결 함수 ====================
def get_conn():
    """DB 연결 생성"""
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT")
    )


# ==================== DB 체크 함수 ====================
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
    """DB 통계 가져오기"""
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                # 총 문서 수
                cur.execute("SELECT COUNT(*) FROM crawled_data;")
                total_docs = cur.fetchone()[0]
                
                # 고유 제목 수
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


# ==================== 전체 DB 준비 ====================
def prepare_all_databases():
    """모든 DB 테이블 생성"""
    print("\n" + "="*60)
    print("===== DB 초기화 시작 =====")
    print("="*60)
    
    try:
        # 1. 웹 DB 생성
        if not table_exists("crawled_data"):
            print("\n🌐 [1/3] 웹 DB 생성 중...")
            if os.path.exists("./extracted_texts"):
                create_db("./extracted_texts")
                print("✅ 웹 DB 생성 완료")
            else:
                print("⚠️ ./extracted_texts 폴더가 없습니다. 웹 DB 건너뛰기")
        else:
            print("✅ [1/3] 웹 DB 이미 존재")
        
        # 2. 책 DB 생성
        if not table_exists("book_eng"):
            print("\n📚 [2/3] 책 DB 생성 중...")
            create_book_db()
            print("✅ 책 DB 생성 완료")
        else:
            print("✅ [2/3] 책 DB 이미 존재")
        
        # 3. 영상 DB 생성
        if not table_exists("video_db"):
            print("\n🎬 [3/3] 영상 DB 생성 중...")
            if os.path.exists("./srt_data"):
                create_video_db_from_folder("./srt_data")
                print("✅ 영상 DB 생성 완료")
            else:
                print("⚠️ ./srt_data 폴더가 없습니다. 영상 DB 건너뛰기")
        else:
            print("✅ [3/3] 영상 DB 이미 존재")
        
        print("\n" + "="*60)
        print("===== DB 초기화 완료 =====")
        print("="*60 + "\n")
        return True
        
    except Exception as e:
        print(f"\n❌ DB 생성 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


# ==================== 사이드바 ====================
with st.sidebar:
    st.markdown(
        '<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQtl4elGepo51WtnezrB1eiyE0TS2GGejnufA&s" '
        'style="display:block; margin:auto; width:150px; border-radius:10px;">',
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<h3 style='text-align: center;'>🤖 BeBot - Let there be light 🤖</h3>",
        unsafe_allow_html=True
    )
    st.markdown("---")
    
    # ==================== DB 자동 초기화 ====================
    if not st.session_state.db_ready:
        # 자동으로 DB 확인 및 생성
        with st.spinner("🔍 DB 확인 중..."):
            web_exists = table_exists("crawled_data")
            if(web_exists): print("✅ [1/3] 웹 DB 존재 확인 완료")
            book_exists = table_exists("book_eng")
            if(book_exists): print("✅ [2/3] 책 DB 존재 확인 완료")
            video_exists = table_exists("video_db")
            if(video_exists): print("✅ [3/3] 영상 DB 존재 확인 완료")
            
            # 하나라도 없으면 자동 생성
            if not (web_exists and book_exists and video_exists):
                st.info("⚙️ DB 초기화 중... (최초 1회, 수 분 소요)")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("DB 생성 중...")
                progress_bar.progress(50)
                
                success = prepare_all_databases()
                
                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()
                
                if success:
                    st.session_state.db_ready = True
                    st.success("✅ DB 준비 완료!")
                    st.rerun()
                else:
                    st.error("❌ DB 생성 실패. 로그를 확인하세요.")
                    st.stop()
            else:
                # 모든 테이블이 이미 존재
                st.session_state.db_ready = True
                st.rerun()

    else:
        st.success("✅ DB 사용 가능")

        # DB 통계
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
        help="LangGraph 실행 과정을 실시간으로 표시합니다"
    )
    
    temperature = st.slider(
        "창의성 수준",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="높을수록 더 창의적이지만 덜 정확할 수 있습니다"
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
    
    # 그래프 시각화
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
st.markdown('<div class="sub-title">Ask questions about the biblical worldview and creation science</div>', unsafe_allow_html=True)

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

# 이전 메시지 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # 영상 출처가 있으면 답변 후 표시
        if msg["role"] == "assistant" and "video_source" in msg and msg["video_source"]:
            video_info = msg["video_source"]
            st.markdown("---")
            st.markdown(f"### 🎬 참고 영상")
            
            # YouTube 영상 임베드
            if "youtu.be" in video_info['url'] or "youtube.com" in video_info['url']:
                # URL에서 비디오 ID 추출
                if "youtu.be" in video_info['url']:
                    video_id = video_info['url'].split('youtu.be/')[-1].split('?')[0]
                else:
                    video_id = video_info['url'].split('v=')[-1].split('&')[0]
                
                # 시작 시간 파라미터 추가
                start_time = int(video_info['start'])
                embed_url = f"https://www.youtube.com/embed/{video_id}?start={start_time}"
                
                st.markdown(f"**{video_info['title']}**")
                st.markdown(f'<iframe width="100%" height="400" src="{embed_url}" frameborder="0" allowfullscreen></iframe>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f"**{video_info['title']}**")
                st.markdown(f"[영상 보기]({video_info['url']}) (시작: {int(video_info['start'])}초)")
        
        # 워크플로우 정보가 있으면 표시
        if "workflow" in msg and st.session_state.show_workflow:
            with st.expander("🔍 처리 과정 보기"):
                for step in msg["workflow"]:
                    st.markdown(f'<div class="workflow-step">✓ {step}</div>', unsafe_allow_html=True)


# 사용자 입력 처리
prompt = st.chat_input("Curious about creation science ✨") or example_prompt

if prompt:
    
    # DB 준비 확인
    if not st.session_state.db_ready:
        st.error("⚠️ DB가 준비되지 않았습니다. 사이드바에서 'DB 생성하기' 버튼을 눌러주세요.")
        st.stop()
    
    # 사용자 메시지 출력
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # AI 응답 생성
    with st.chat_message("assistant"):
        workflow_steps = []
        
        # 워크플로우 단계 표시 컨테이너
        if st.session_state.show_workflow:
            workflow_container = st.empty()
        
        # 응답 생성
        with st.spinner("Searching..."):
            try:
                response, sources_info = generate(prompt)
                
                # 워크플로우 단계 기록
                workflow_steps = [
                    "질문 라우팅 완료",
                    "벡터 검색 수행 (웹 + 책 + 영상)",
                    "문서 적합성 판단",
                    "답변 생성 완료"
                ]
                
                # 워크플로우 표시
                if st.session_state.show_workflow:
                    with workflow_container:
                        with st.expander("🔍 Processing", expanded=True):
                            for step in workflow_steps:
                                st.markdown(f'<div class="workflow-step">✓ {step}</div>', unsafe_allow_html=True)
                
                # 응답 출력
                st.markdown(response)
                
                # 영상 출처가 있으면 답변 후 표시
                video_source = None
                if sources_info.get("video_docs"):
                    top_video = sources_info["video_docs"][0]
                    video_source = {
                        "title": top_video["title"],
                        "url": top_video["url"],
                        "start": top_video["start"]
                    }
                    
                    st.markdown("---")
                    st.markdown(f"### 🎬 참고 영상")
                    
                    # YouTube 영상 임베드
                    if "youtu.be" in top_video['url'] or "youtube.com" in top_video['url']:
                        # URL에서 비디오 ID 추출
                        if "youtu.be" in top_video['url']:
                            video_id = top_video['url'].split('youtu.be/')[-1].split('?')[0]
                        else:
                            video_id = top_video['url'].split('v=')[-1].split('&')[0]
                        
                        # 시작 시간 파라미터 추가
                        start_time = int(top_video['start'])
                        embed_url = f"https://www.youtube.com/embed/{video_id}?start={start_time}"
                        
                        st.markdown(f"**{top_video['title']}**")
                        st.markdown(f'<iframe width="100%" height="400" src="{embed_url}" frameborder="0" allowfullscreen></iframe>', 
                                   unsafe_allow_html=True)
                    else:
                        st.markdown(f"**{top_video['title']}**")
                        st.markdown(f"[영상 보기]({top_video['url']}) (시작: {int(top_video['start'])}초)")
                
                # 출처 추가
                sources = []
                
                if sources_info.get("web_docs"):
                    web_urls = list(set([d["url"] for d in sources_info["web_docs"] if d.get("url")]))
                    if web_urls:
                        sources.append("\n**🌐 웹 자료:**")
                        for url in web_urls:
                            sources.append(f"• {url}")
                
                if sources_info.get("book_docs"):
                    book_names = list(set([d['book'] for d in sources_info["book_docs"]]))
                    if len(book_names) == 1:
                        book_name = book_names[0]
                        pages = ", ".join(str(d['page']) for d in sources_info["book_docs"])
                        sources.append(f"\n**📖 책 자료:**")
                        sources.append(f"• {book_name} - 페이지 {pages}")
                    else:
                        sources.append(f"\n**📖 책 자료:**")
                        for doc in sources_info["book_docs"]:
                            sources.append(f"• {doc['book']} - 페이지 {doc['page']}")
                
                if sources_info.get("video_docs"):
                    sources.append("\n**🎬 영상 자료:**")
                    for doc in sources_info["video_docs"]:
                        sources.append(f"• [{doc['title']}]({doc['url']}) - {int(doc['start'])}초~{int(doc['end'])}초")
                
                if sources:
                    st.markdown("\n---\n" + "\n".join(sources))
                
            except Exception as e:
                st.error(f"❌ 오류 발생: {str(e)}")
                response = "죄송합니다. 답변 생성 중 오류가 발생했습니다."
                workflow_steps = []
                video_source = None
    
    # 메시지 저장 (워크플로우 정보 포함)
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "workflow": workflow_steps,
        "video_source": video_source
    })


# ==================== 푸터 ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>🙏 All answers are based on a biblical perspective</p>
    <p>📚 Source: Korean Creation Science Association & related sources</p>
</div>
""", unsafe_allow_html=True)