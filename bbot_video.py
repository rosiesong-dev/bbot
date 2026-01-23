# bbot_video.py
import os
import asyncio
import psycopg2
from psycopg2.extras import execute_values
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import pysrt

load_dotenv()

# ==================== 모델 초기화 ====================
api_key = os.getenv("UPSTAGE_API_KEY")
base_url = os.getenv("UPSTAGE_BASE_URL")

llm_model = ChatUpstage(api_key=api_key, base_url=base_url)
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

# ==================== Pydantic Model ====================
class FilteredContent(BaseModel):
    category: str = Field(description="Intro | Outro | Content | Mixed | Noise")
    reasoning: str
    is_useful: bool
    cleaned_text: str

# ==================== Chunking Utilities ====================
semantic_chunker = SemanticChunker(
    embeddings=embedding_model,
    breakpoint_threshold_type="gradient",
)

force_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50,
    separators=[".", " ", ""]
)

# ==================== SRT 처리 함수 ====================
def parse_srt(file_path: str) -> List[Dict]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    subs = pysrt.open(file_path, encoding='utf-8')  # ✅ encoding 추가
    results = []
    for sub in subs:
        text = sub.text.replace("\n", " ").strip()
        if not text or (text.startswith("[") and text.endswith("]")):
            continue
        results.append({
            "start": sub.start.ordinal / 1000.0,
            "end": sub.end.ordinal / 1000.0,
            "text": text
        })
    print(f"📄 SRT 파싱 완료: {len(results)}개 자막")
    return results


def force_split_chunk(huge_chunk: Dict) -> List[Dict]:
    full_text = huge_chunk["content"]
    start_t = huge_chunk["start_time"]
    end_t = huge_chunk["end_time"]
    total_duration = end_t - start_t
    total_len = len(full_text)

    sub_texts = force_splitter.split_text(full_text)

    result_chunks = []
    current_t = start_t

    for text_part in sub_texts:
        part_len = len(text_part)
        ratio = part_len / total_len if total_len > 0 else 0
        duration_part = total_duration * ratio

        result_chunks.append({
            "start_time": current_t,
            "end_time": current_t + duration_part,
            "content": text_part
        })
        current_t += duration_part

    return result_chunks


def semantic_chunking_with_safety_merge(srt_items: List[Dict]) -> List[Dict]:
    print("✂️ Semantic chunking...")
    sentences = [item["text"] for item in srt_items]
    if not sentences:
        return []

    raw_chunks = semantic_chunker.split_text("\n".join(sentences))
    print(f"📦 Semantic chunker 결과: {len(raw_chunks)}개 청크")

    aligned_chunks = []
    cursor = 0

    for chunk_text in raw_chunks:
        chunk_sentences = [s.strip() for s in chunk_text.split("\n") if s.strip()]
        if not chunk_sentences:
            continue

        chunk_len = len(chunk_sentences)
        start_idx = min(cursor, len(srt_items) - 1)
        end_idx = min(cursor + chunk_len - 1, len(srt_items) - 1)

        start_item = srt_items[start_idx]
        end_item = srt_items[end_idx]

        primary_chunk = {
            "start_time": start_item["start"],
            "end_time": end_item["end"],
            "content": " ".join(chunk_sentences)
        }

        if len(primary_chunk["content"]) > 1200:
            aligned_chunks.extend(force_split_chunk(primary_chunk))
        else:
            aligned_chunks.append(primary_chunk)

        cursor += chunk_len

    print(f"📦 최종 청크 수: {len(aligned_chunks)}개")
    return aligned_chunks

# ==================== LLM 정제 ====================
async def refine_chunk_async(chunk: Dict, chain) -> Optional[Dict]:
    try:
        res = await chain.ainvoke({"text": chunk["content"]})
        if res and res.is_useful and res.cleaned_text.strip():
            if len(res.cleaned_text) < 10:
                return None
            chunk["content"] = res.cleaned_text
            return chunk
    except Exception as e:
        print(f"⚠️ LLM 처리 오류: {e}")
    return None


async def process_srt_file(file_path: str) -> List[Dict]:
    srt_items = parse_srt(file_path)
    
    if not srt_items:
        print("⚠️ 파싱된 자막이 없습니다")
        return []
    
    raw_chunks = semantic_chunking_with_safety_merge(srt_items)

    if not raw_chunks:
        print("⚠️ 청크가 생성되지 않았습니다")
        return []

    print(f"🧹 {len(raw_chunks)}개 청크를 LLM으로 정제 중...")

    system_prompt = """
    You are an expert subtitle editor.
    Clean spoken text into proper written English.
    Remove greetings, fillers, and self-introductions.
    Split run-on sentences into proper sentences.
    Do NOT summarize facts.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Input Text: {text}")
    ])

    chain = prompt | llm_model.with_structured_output(FilteredContent)

    tasks = [refine_chunk_async(chunk, chain) for chunk in raw_chunks]
    refined = await asyncio.gather(*tasks)

    final_chunks = [r for r in refined if r is not None]
    print(f"✅ 유효한 청크: {len(final_chunks)}개")
    return final_chunks

# ==================== DB Functions ====================
def create_video_table():
    """video_db 테이블 생성"""
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS video_db (
                        id SERIAL PRIMARY KEY,
                        video_id TEXT NOT NULL,
                        title TEXT,
                        start_time DOUBLE PRECISION,
                        end_time DOUBLE PRECISION,
                        timestamp TIMESTAMP DEFAULT NOW(),
                        url TEXT,
                        content TEXT,
                        content_embedding vector(4096)
                    );
                """)
                conn.commit()
        print("✅ video_db 테이블 생성 완료\n")
    except Exception as e:
        print(f"❌ 테이블 생성 실패: {e}")


def delete_video_data(video_id: str):
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM video_db WHERE video_id = %s", (video_id,))
                deleted_count = cur.rowcount
                conn.commit()
                if deleted_count > 0:
                    print(f"🗑️ 기존 데이터 {deleted_count}개 삭제됨")
    except Exception as e:
        print(f"❌ 데이터 삭제 실패: {e}")


def save_video_segments(data_tuples: List[Tuple]):
    if not data_tuples:
        print("⚠️ 저장할 데이터가 없습니다")
        return
    
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                execute_values(cur, """
                    INSERT INTO video_db
                    (video_id, title, start_time, end_time, url, content, content_embedding)
                    VALUES %s
                """, data_tuples)
                conn.commit()
        print(f"✅ {len(data_tuples)}개 세그먼트 저장 완료")
    except Exception as e:
        print(f"❌ 데이터 저장 실패: {e}")
        import traceback
        traceback.print_exc()


# ==================== 파일명에서 메타데이터 추출 ====================
def extract_video_meta_from_filename(srt_path: str, idx: int) -> dict:
    """
    SRT 파일명에서 video_id, title, url 추출
    """
    filename = os.path.basename(srt_path).replace(".srt", "")
    
    # "_"로 분리 시도
    if "_" in filename:
        parts = filename.rsplit("_", 1)
        if len(parts) == 2:
            title_part, id_part = parts
            # .en 같은 확장자 제거
            youtube_id = id_part.split(".")[0]
        else:
            title_part = filename
            youtube_id = ""
    else:
        title_part = filename
        youtube_id = ""
    
    # DB용 video_id 생성
    video_id = f"v{idx:04d}"
    title = title_part.strip()
    url = f"https://youtu.be/{youtube_id}" if youtube_id else ""
    
    return {
        "video_id": video_id,
        "title": title,
        "url": url
    }


# ==================== 단일 영상 처리 ====================
async def process_single_video(srt_path: str, video_meta: dict):
    """단일 SRT 파일 처리"""
    print(f"\n{'='*60}")
    print(f"🎬 처리 중: {video_meta['title']}")
    print(f"   Video ID: {video_meta['video_id']}")
    print(f"   URL: {video_meta['url']}")
    print(f"{'='*60}")
    
    try:
        refined_chunks = await process_srt_file(srt_path)
    except Exception as e:
        print(f"❌ SRT 처리 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if not refined_chunks:
        print(f"⚠️ {video_meta['title']}: 유효한 청크 없음\n")
        return

    print(f"🔮 {len(refined_chunks)}개 청크 임베딩 생성 중...")
    texts = [c["content"] for c in refined_chunks]
    
    try:
        vectors = embedding_model.embed_documents(texts)
        print(f"✅ 임베딩 생성 완료")
    except Exception as e:
        print(f"❌ 임베딩 생성 실패: {e}")
        return

    # 기존 데이터 삭제
    delete_video_data(video_meta["video_id"])

    # 데이터 준비
    rows = []
    for i, chunk in enumerate(refined_chunks):
        time_url = f"{video_meta['url']}?t={int(chunk['start_time'])}s"
        rows.append((
            video_meta["video_id"],
            video_meta["title"],
            chunk["start_time"],
            chunk["end_time"],
            time_url,
            chunk["content"],
            vectors[i]
        ))

    # DB 저장
    save_video_segments(rows)
    print(f"✅ {video_meta['title']}: 처리 완료\n")


# ==================== 폴더 전체 처리 ====================
def create_video_db_from_folder(srt_folder: str):
    """SRT 폴더의 모든 파일 처리"""
    print(f"\n{'='*60}")
    print(f"===== 영상 DB 생성 시작 =====")
    print(f"{'='*60}\n")
    
    # 1. 테이블 생성
    create_video_table()
    
    # 2. SRT 파일 목록
    if not os.path.exists(srt_folder):
        print(f"⚠️ 폴더가 없습니다: {srt_folder}")
        return
    
    srt_files = sorted([f for f in os.listdir(srt_folder) if f.endswith(".srt")])
    
    if not srt_files:
        print(f"⚠️ SRT 파일이 없습니다: {srt_folder}")
        return
    
    print(f"📁 {len(srt_files)}개 SRT 파일 발견\n")
    
    # 3. 각 파일 처리
    for idx, srt_file in enumerate(srt_files, 1):
        print(f"\n[{idx}/{len(srt_files)}] {srt_file}")
        
        srt_path = os.path.join(srt_folder, srt_file)
        video_meta = extract_video_meta_from_filename(srt_path, idx)
        
        try:
            asyncio.run(process_single_video(srt_path, video_meta))
        except Exception as e:
            print(f"❌ {srt_file} 처리 실패: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"===== 영상 DB 생성 완료 =====")
    print(f"{'='*60}\n")



# ==================== 영상 검색 함수 ====================
def retrieve_video_segments(question: str, top_k: int = 3):
    """영상 세그먼트 벡터 검색"""
    print(f"\n🎬 [Video] 영상 검색 중: {question}")

    q_emb = embedding_model.embed_query(question)
    print("🧠 질문 임베딩 생성 완료")

    with get_conn() as conn:
        with conn.cursor() as cur:
            # video_db 테이블이 있는지 확인
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'video_db'
                );
            """)
            
            if not cur.fetchone()[0]:
                print("⚠️ video_db 테이블이 없습니다.")
                return []
            
            cur.execute("""
                SELECT
                    video_id,
                    title,
                    start_time,
                    end_time,
                    url,
                    content
                FROM video_db
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


# ==================== 실행부 ====================
if __name__ == "__main__":
    SRT_FOLDER = "./srt_data"
    create_video_db_from_folder(SRT_FOLDER)

    