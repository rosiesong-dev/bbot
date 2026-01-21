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

    subs = pysrt.open(file_path)
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
        print(f"Processing error: {e}")
    return None


async def process_srt_file(file_path: str) -> List[Dict]:
    srt_items = parse_srt(file_path)
    raw_chunks = semantic_chunking_with_safety_merge(srt_items)

    if not raw_chunks:
        return []

    print(f"🧹 Refining {len(raw_chunks)} chunks in parallel...")

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
    print(f"✅ Final valid chunks: {len(final_chunks)}")
    return final_chunks

# ==================== DB Functions ====================
def create_video_table():
    """video_segments 테이블만 생성"""
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
    print("✅ video_db 테이블 생성 완료")


def delete_video_data(video_id: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM video_db WHERE video_id = %s", (video_id,))
            conn.commit()


def save_video_segments(data_tuples: List[Tuple]):
    with get_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO video_db
                (video_id, title, start_time, end_time, url, content, content_embedding)
                VALUES %s
            """, data_tuples)
            conn.commit()


# ==================== 파일명에서 메타데이터 추출 ====================
def extract_video_meta_from_filename(srt_path: str) -> dict:
    """
    SRT 파일명에서 video_id, title, url 추출
    예: "video123_강의제목.srt" → {"video_id": "video123", "title": "강의제목", ...}
    """
    filename = os.path.basename(srt_path).replace(".srt", "")
    
    # 간단한 파싱 로직 (필요시 수정)
    parts = filename.split("_", 1)
    
    if len(parts) >= 2:
        video_id = parts[0]
        title = parts[1]
    else:
        video_id = filename
        title = filename
    
    # YouTube URL 생성 (실제로는 별도 메타데이터 파일이나 DB에서 가져와야 함)
    url = f"https://youtu.be/{video_id}"
    
    return {
        "video_id": video_id,
        "title": title,
        "url": url
    }


# ==================== 단일 영상 처리 ====================
async def process_single_video(srt_path: str, video_meta: dict):
    """단일 SRT 파일 처리"""
    print(f"\n🎬 처리 중: {video_meta['title']}")
    
    refined_chunks = await process_srt_file(srt_path)
    if not refined_chunks:
        print(f"⚠️ {video_meta['title']}: 유효한 청크 없음")
        return

    texts = [c["content"] for c in refined_chunks]
    vectors = embedding_model.embed_documents(texts)

    delete_video_data(video_meta["video_id"])

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

    save_video_segments(rows)
    print(f"✅ {video_meta['title']}: {len(rows)}개 세그먼트 저장 완료")


# ==================== 폴더 전체 처리 ====================
def create_video_db_from_folder(srt_folder: str):
    """SRT 폴더의 모든 파일 처리"""
    print(f"\n{'='*60}")
    print(f"===== 영상 DB 생성 시작 =====")
    print(f"{'='*60}")
    
    # 1. 테이블 생성
    create_video_table()
    
    # 2. SRT 파일 목록
    if not os.path.exists(srt_folder):
        print(f"⚠️ 폴더가 없습니다: {srt_folder}")
        return
    
    srt_files = [f for f in os.listdir(srt_folder) if f.endswith(".srt")]
    
    if not srt_files:
        print(f"⚠️ SRT 파일이 없습니다: {srt_folder}")
        return
    
    print(f"📁 {len(srt_files)}개 SRT 파일 발견")
    
    # 3. 각 파일 처리
    for idx, srt_file in enumerate(srt_files, 1):
        print(f"\n[{idx}/{len(srt_files)}] 처리 중: {srt_file}")
        
        srt_path = os.path.join(srt_folder, srt_file)
        video_meta = extract_video_meta_from_filename(srt_path)
        
        try:
            asyncio.run(process_single_video(srt_path, video_meta))
        except Exception as e:
            print(f"❌ {srt_file} 처리 실패: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"===== 영상 DB 생성 완료 =====")
    print(f"{'='*60}\n")