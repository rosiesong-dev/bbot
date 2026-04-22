# DB 초기화 전용 — python db_init.py 로 최초 1회 실행
import pandas as pd
import os
import asyncio
import pdfplumber
import pysrt
import tiktoken
from datetime import datetime
from typing import List, Dict, Optional
from psycopg2.extras import execute_values
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import unicodedata

from config import get_conn, EMBED_DIM
from llm_factory import get_embedding, get_llm

embedding_model = get_embedding()
llm_model       = get_llm()

# ==================== 경로 설정 ====================
BOOKS_FOLDER = "./books"
WEB_FOLDER = "./extracted_texts"
SRT_FOLDER = "./srt_data"


# ==================== 공통 ====================
def table_exists(table_name: str) -> bool:
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables
                        WHERE table_schema = 'public' AND table_name = %s
                    );
                """, (table_name,))
                return cur.fetchone()[0]
    except Exception as e:
        print(f"❌ 테이블 확인 에러 ({table_name}): {e}")
        return False


# ==================== [1] 웹 DB ====================
enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

def split_text_by_tokens(text: str, max_tokens: int = 4000):
    words, chunks, chunk, total = text.split(), [], [], 0
    for word in words:
        wt = count_tokens(word + " ")
        if total + wt > max_tokens:
            chunks.append(" ".join(chunk))
            chunk, total = [], 0
        chunk.append(word)
        total += wt
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks


def create_web_db(folder_path: str, max_tokens: int = 4000):
    print("\n📁 [1/3] 웹 DB 생성 시작...")

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS crawled_data (
                    id SERIAL PRIMARY KEY,
                    title TEXT, url TEXT, crawl_time TIMESTAMP,
                    content TEXT, content_embedding vector({EMBED_DIM})
                );
            """)
            conn.commit()

    files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    print(f"   총 {len(files)}개 파일 발견")
    inserted, failed = 0, []

    with get_conn() as conn:
        with conn.cursor() as cur:
            for idx, fname in enumerate(files, 1):
                try:
                    with open(os.path.join(folder_path, fname), "r", encoding="utf-8") as f:
                        full_text = f.read()

                    lines = full_text.split('\n')
                    title, url, crawl_time, content_start = "", "", None, 0
                    for i, line in enumerate(lines):
                        if line.startswith("Title:"):       title = line[6:].strip()
                        elif line.startswith("URL:"):        url   = line[4:].strip()
                        elif line.startswith("Crawl Time:"):
                            try: crawl_time = datetime.fromisoformat(line[11:].strip().replace('+09:00',''))
                            except: crawl_time = datetime.now()
                        elif line.startswith("Content:"):   content_start = i + 1; break

                    content = '\n'.join(lines[content_start:]).strip() if content_start else full_text
                    if not title: title = fname.replace(".txt", "")

                    for chunk in split_text_by_tokens(content, max_tokens):
                        vec = embedding_model.embed_query(chunk)
                        cur.execute(
                            "INSERT INTO crawled_data (title, url, crawl_time, content, content_embedding) VALUES (%s,%s,%s,%s,%s::vector)",
                            (title, url, crawl_time, chunk, vec)
                        )
                        inserted += 1

                    if idx % 50 == 0:
                        conn.commit()
                        print(f"   {idx}/{len(files)} 완료...")

                except Exception as e:
                    print(f"   ⚠️ 실패: {fname} → {e}")
                    failed.append(fname)
                    conn.rollback()
            conn.commit()

    print(f"✅ 웹 DB 완료 (삽입: {inserted}개, 실패: {len(failed)}개)\n")


# ==================== [2] 책 DB ====================
def create_book_db():
    print("\n📚 [2/3] 책 DB 생성 시작...")

    for lang in ["en", "ko"]:
        folder_path = os.path.join(BOOKS_FOLDER, lang)

        if not os.path.exists(folder_path):
            print(f"   ⚠️ {folder_path} 없음 → 스킵")
            continue

        table_name = f"book_{lang}"

        # 테이블 생성
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id SERIAL PRIMARY KEY,
                        book_name TEXT,
                        page_num INT,
                        content TEXT,
                        embedding vector({EMBED_DIM})
                    );
                """)
                conn.commit()

        pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
        print(f"\n📂 [{lang}] 총 {len(pdf_files)}개 PDF")

        for pdf_file in pdf_files:
            pdf_path = os.path.join(folder_path, pdf_file)
            book_name = pdf_file.replace(".pdf", "")

            # 중복 체크
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT COUNT(*) FROM {table_name} WHERE book_name = %s", (book_name,))
                    if cur.fetchone()[0] > 0:
                        print(f"   '{book_name}' 이미 존재 → 스킵")
                        continue

            try:
                with pdfplumber.open(pdf_path) as pdf:
                    total, inserted = len(pdf.pages), 0
                    print(f"   📖 {book_name} ({total} 페이지)")

                    with get_conn() as conn:
                        with conn.cursor() as cur:
                            for i, page in enumerate(pdf.pages, 1):
                                text = page.extract_text()

                                if not text or len(text.strip()) < 50:
                                    continue

                                # 🔥 핵심: embedding 실패해도 계속 진행
                                try:
                                    vec = embedding_model.embed_query(text)
                                except Exception as e:
                                    print(f"      ⚠️ embedding 실패 (page {i})")
                                    vec = [0.0] * EMBED_DIM

                                cur.execute(f"""
                                    INSERT INTO {table_name} (book_name, page_num, content, embedding)
                                    VALUES (%s, %s, %s, %s::vector)
                                """, (book_name, i, text, vec))

                                inserted += 1

                                if i % 10 == 0:
                                    print(f"      페이지 {i}/{total}...")

                            conn.commit()

                print(f"   ✅ {book_name} 완료 ({inserted}페이지)")
    

            except Exception as e:
                print(f"   ❌ {book_name} 처리 실패 → {e}")

        if lang == "ko":
            excel_path = f"./image_data/{book_name}.xlsx"
            insert_images_from_excel(excel_path, book_name)


# ==================== [3] 영상 DB ====================
class FilteredContent(BaseModel):
    category:     str  = Field(description="Intro | Outro | Content | Mixed | Noise")
    reasoning:    str
    is_useful:    bool
    cleaned_text: str


semantic_chunker = SemanticChunker(embeddings=embedding_model, breakpoint_threshold_type="gradient")
force_splitter   = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50, separators=[".", " ", ""])


def parse_srt(file_path: str) -> List[Dict]:
    subs = pysrt.open(file_path, encoding='utf-8')
    results = []
    for sub in subs:
        text = sub.text.replace("\n", " ").strip()
        if not text or (text.startswith("[") and text.endswith("]")): continue
        results.append({"start": sub.start.ordinal/1000.0, "end": sub.end.ordinal/1000.0, "text": text})
    return results


def force_split_chunk(chunk: Dict) -> List[Dict]:
    full, start_t, end_t = chunk["content"], chunk["start_time"], chunk["end_time"]
    total_dur, total_len = end_t - start_t, len(full)
    result, current = [], start_t
    for part in force_splitter.split_text(full):
        ratio = len(part) / total_len if total_len else 0
        result.append({"start_time": current, "end_time": current + total_dur * ratio, "content": part})
        current += total_dur * ratio
    return result


def semantic_chunking(srt_items: List[Dict]) -> List[Dict]:
    sentences = [item["text"] for item in srt_items]
    if not sentences: return []
    raw_chunks = semantic_chunker.split_text("\n".join(sentences))
    aligned, cursor = [], 0
    for chunk_text in raw_chunks:
        chunk_sentences = [s.strip() for s in chunk_text.split("\n") if s.strip()]
        if not chunk_sentences: continue
        s_idx = min(cursor, len(srt_items)-1)
        e_idx = min(cursor + len(chunk_sentences)-1, len(srt_items)-1)
        primary = {"start_time": srt_items[s_idx]["start"], "end_time": srt_items[e_idx]["end"],
                   "content": " ".join(chunk_sentences)}
        aligned.extend(force_split_chunk(primary) if len(primary["content"]) > 1200 else [primary])
        cursor += len(chunk_sentences)
    return aligned


async def refine_chunk(chunk: Dict, chain) -> Optional[Dict]:
    try:
        res = await chain.ainvoke({"text": chunk["content"]})
        if res and res.is_useful and len(res.cleaned_text.strip()) >= 10:
            chunk["content"] = res.cleaned_text
            return chunk
    except Exception as e:
        print(f"   ⚠️ 정제 오류: {e}")
    return None


async def process_srt_file(file_path: str) -> List[Dict]:
    srt_items  = parse_srt(file_path)
    if not srt_items: return []
    raw_chunks = semantic_chunking(srt_items)
    if not raw_chunks: return []

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert subtitle editor. Clean spoken text into proper written English. "
                   "Remove greetings, fillers, self-introductions. Do NOT summarize facts."),
        ("human", "Input Text: {text}")
    ])
    chain   = prompt | llm_model.with_structured_output(FilteredContent)
    refined = await asyncio.gather(*[refine_chunk(c, chain) for c in raw_chunks])
    return [r for r in refined if r is not None]


def extract_video_meta(srt_path: str, idx: int) -> dict:
    filename = os.path.basename(srt_path).replace(".srt", "")
    if "_" in filename:
        parts      = filename.rsplit("_", 1)
        title_part = parts[0].strip()
        youtube_id = parts[1].split(".")[0] if len(parts) == 2 else ""
    else:
        title_part, youtube_id = filename, ""
    return {"video_id": f"v{idx:04d}", "title": title_part,
            "url": f"https://youtu.be/{youtube_id}" if youtube_id else ""}


async def process_single_video(srt_path: str, meta: dict):
    print(f"   🎬 {meta['title']}")
    chunks = await process_srt_file(srt_path)
    if not chunks:
        print("   ⚠️ 유효한 청크 없음"); return

    vectors = embedding_model.embed_documents([c["content"] for c in chunks])

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM video_db WHERE video_id = %s", (meta["video_id"],))
            conn.commit()

    rows = [(meta["video_id"], meta["title"], c["start_time"], c["end_time"],
             f"{meta['url']}?t={int(c['start_time'])}s", c["content"], vectors[i])
            for i, c in enumerate(chunks)]

    with get_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO video_db (video_id, title, start_time, end_time, url, content, content_embedding)
                VALUES %s
            """, rows)
            conn.commit()
    print(f"   ✅ {len(rows)}개 세그먼트 저장")


def create_video_db(srt_folder: str):
    print("\n🎬 [3/3] 영상 DB 생성 시작...")

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS video_db (
                    id SERIAL PRIMARY KEY,
                    video_id TEXT NOT NULL, title TEXT,
                    start_time DOUBLE PRECISION, end_time DOUBLE PRECISION,
                    timestamp TIMESTAMP DEFAULT NOW(),
                    url TEXT, content TEXT,
                    content_embedding vector({EMBED_DIM})
                );
            """)
            conn.commit()

    srt_files = sorted([f for f in os.listdir(srt_folder) if f.endswith(".srt")])
    if not srt_files:
        print("   ⚠️ SRT 파일 없음"); return

    print(f"   총 {len(srt_files)}개 SRT 파일")
    for idx, srt_file in enumerate(srt_files, 1):
        print(f"\n   [{idx}/{len(srt_files)}] {srt_file}")
        meta = extract_video_meta(os.path.join(srt_folder, srt_file), idx)
        try:
            asyncio.run(process_single_video(os.path.join(srt_folder, srt_file), meta))
        except Exception as e:
            print(f"   ❌ 실패: {e}")

    print("✅ 영상 DB 완료\n")

# ==================== 이미지 DB ====================
def create_image_table():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS book_images (
                    id SERIAL PRIMARY KEY,
                    book_name TEXT NOT NULL,
                    page_num INT NOT NULL,
                    file_name TEXT,
                    file_path TEXT
                );
            """)
            conn.commit()
    print("🖼️ book_images 테이블 준비 완료")


def insert_images_from_excel(excel_path, book_name):
    if not os.path.exists(excel_path):
        print(f"⚠️ 이미지 엑셀 없음 → {excel_path}")
        return

    df = pd.read_excel(excel_path)
    print(f"🖼️ {book_name} 이미지 {len(df)}개 발견")

    inserted = 0

    with get_conn() as conn:
        with conn.cursor() as cur:
            for _, row in df.iterrows():
                try:
                    page_num = int(row.get("page", row.get("page_num")))
                    file_name = row["file_name"]
                    file_path = str(row["file_path"]).replace("\\", "/")

                    # jpx 스킵 (선택)
                    if "original_format" in row and row["original_format"] == "jpx":
                        continue

                    cur.execute("""
                        INSERT INTO book_images (book_name, page_num, file_name, file_path)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                    """, (book_name, page_num, file_name, file_path))

                    inserted += 1

                except Exception as e:
                    print(f"   ⚠️ 이미지 insert 실패 → {e}")

        conn.commit()

    print(f"✅ {book_name} 이미지 {inserted}개 저장 완료")


# ==================== 전체 초기화 ====================
def init_all():
    print("\n" + "="*60)
    print("===== BeBot DB 초기화 시작 =====")
    print("="*60)

    
    create_image_table()

    if table_exists("crawled_data"):
        print("✅ [1/4] 웹 DB 이미 존재 → 스킵")
    elif os.path.exists(WEB_FOLDER):
        create_web_db(WEB_FOLDER)
    else:
        print(f"⚠️ [1/4] {WEB_FOLDER} 없음 → 스킵")

    if os.path.exists(BOOKS_FOLDER):
        create_book_db()
    else:
        print("⚠️ books 폴더 없음")



    if table_exists("video_db"):
        print("✅ [4/4] 영상 DB 이미 존재 → 스킵")
    elif os.path.exists(SRT_FOLDER):
        create_video_db(SRT_FOLDER)
    else:
        print(f"⚠️ [4/4] {SRT_FOLDER} 없음 → 스킵")

    print("\n" + "="*60)
    print("===== DB 초기화 완료 =====")
    print("="*60 + "\n")


if __name__ == "__main__":
    init_all()