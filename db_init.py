# DB 초기화 전용 — python db_init.py

import os
import asyncio
import pandas as pd
import pdfplumber
import pysrt
import tiktoken
import unicodedata

from datetime import datetime
from typing import List, Dict, Optional
from psycopg2.extras import execute_values

from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config import get_conn, EMBED_DIM, PROVIDER
from llm_factory import get_embedding, get_llm


# ==================== Models ====================
embedding_model = get_embedding()
llm_model = get_llm()


# ==================== Embedding Layer ====================

def safe_text(text: str, limit: int = 3000):
    if not text:
        return None
    text = unicodedata.normalize("NFKC", text)
    return text.strip()[:limit]


def get_text_embedding(text: str):
    text = safe_text(text)
    if not text:
        return [0.0] * EMBED_DIM

    if PROVIDER == "gemma":
        return embedding_model.encode(text).tolist()

    return embedding_model.embed_query(text)


def get_batch_embeddings(texts):
    cleaned = [safe_text(t) for t in texts]
    cleaned = [t for t in cleaned if t]

    if not cleaned:
        return []

    if PROVIDER == "gemma":
        return embedding_model.encode(cleaned, show_progress_bar=False).tolist()

    return embedding_model.embed_documents(cleaned)


# ==================== Paths ====================
BOOKS_FOLDER = "./books"
WEB_FOLDER = "./extracted_texts"
SRT_FOLDER = "./srt_data"


# ==================== Utils ====================

enc = tiktoken.get_encoding("cl100k_base")

def split_text_by_tokens(text: str, max_tokens: int = 4000):
    words = text.split()
    chunks, chunk, total = [], [], 0

    for w in words:
        wt = len(enc.encode(w + " "))
        if total + wt > max_tokens:
            chunks.append(" ".join(chunk))
            chunk, total = [], 0
        chunk.append(w)
        total += wt

    if chunk:
        chunks.append(" ".join(chunk))

    return chunks


def table_exists(table_name: str):
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables
                        WHERE table_schema='public' AND table_name=%s
                    );
                """, (table_name,))
                return cur.fetchone()[0]
    except:
        return False


# ==================== WEB DB ====================

def create_web_db(folder_path):
    print("\n📁 Web DB 생성")

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS crawled_data (
                    id SERIAL PRIMARY KEY,
                    title TEXT,
                    url TEXT,
                    crawl_time TIMESTAMP,
                    content TEXT,
                    content_embedding vector({EMBED_DIM})
                );
            """)
            conn.commit()

    files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

    with get_conn() as conn:
        with conn.cursor() as cur:

            for fname in files:
                try:
                    with open(os.path.join(folder_path, fname), "r", encoding="utf-8") as f:
                        full_text = f.read()

                    lines = full_text.split("\n")
                    title = fname
                    url = ""
                    crawl_time = datetime.now()
                    content = full_text

                    for line in lines:
                        if line.startswith("Title:"):
                            title = line[7:].strip()
                        elif line.startswith("URL:"):
                            url = line[4:].strip()

                    for chunk in split_text_by_tokens(content):

                        vec = get_text_embedding(chunk)

                        cur.execute("""
                            INSERT INTO crawled_data
                            (title, url, crawl_time, content, content_embedding)
                            VALUES (%s,%s,%s,%s,%s::vector)
                        """, (title, url, crawl_time, chunk, vec))

                except Exception as e:
                    print(f"❌ web 실패 {fname}: {e}")

            conn.commit()


# ==================== BOOK DB ====================

def create_book_db():
    print("\n📚 Book DB 생성")

    for lang in ["en", "ko"]:
        path = os.path.join(BOOKS_FOLDER, lang)

        if not os.path.exists(path):
            continue

        table = f"book_{lang}"

        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table} (
                        id SERIAL PRIMARY KEY,
                        book_name TEXT,
                        page_num INT,
                        content TEXT,
                        embedding vector({EMBED_DIM})
                    );
                """)
                conn.commit()

        pdfs = [f for f in os.listdir(path) if f.endswith(".pdf")]

        for pdf in pdfs:
            book_name = pdf.replace(".pdf", "")

            with pdfplumber.open(os.path.join(path, pdf)) as pdf_file:

                with get_conn() as conn:
                    with conn.cursor() as cur:

                        for i, page in enumerate(pdf_file.pages, 1):

                            text = page.extract_text()

                            if not text:
                                continue

                            text = safe_text(text)
                            if not text or len(text) < 30:
                                continue

                            vec = get_text_embedding(text)

                            cur.execute(f"""
                                INSERT INTO {table}
                                (book_name, page_num, content, embedding)
                                VALUES (%s,%s,%s,%s::vector)
                            """, (book_name, i, text, vec))

                        conn.commit()


# ==================== VIDEO DB ====================

class FilteredContent(BaseModel):
    category: str
    reasoning: str
    is_useful: bool
    cleaned_text: str


semantic_chunker = SemanticChunker(
    embeddings=embedding_model,
    breakpoint_threshold_type="gradient"
)

force_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50
)


def parse_srt(path):
    subs = pysrt.open(path)
    return [
        {
            "start": s.start.ordinal / 1000,
            "end": s.end.ordinal / 1000,
            "text": s.text.replace("\n", " ")
        }
        for s in subs
    ]


async def create_video_db(folder):
    print("\n🎬 Video DB 생성")

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS video_db (
                    id SERIAL PRIMARY KEY,
                    video_id TEXT,
                    title TEXT,
                    start_time FLOAT,
                    end_time FLOAT,
                    url TEXT,
                    content TEXT,
                    content_embedding vector({EMBED_DIM})
                );
            """)
            conn.commit()

    files = [f for f in os.listdir(folder) if f.endswith(".srt")]

    for idx, file in enumerate(files):

        path = os.path.join(folder, file)
        subs = parse_srt(path)

        texts = [s["text"] for s in subs]

        vectors = get_batch_embeddings(texts)

        rows = []

        for i, s in enumerate(subs):
            rows.append((
                f"v{idx}",
                file,
                s["start"],
                s["end"],
                "",
                s["text"],
                vectors[i] if i < len(vectors) else [0.0] * EMBED_DIM
            ))

        with get_conn() as conn:
            with conn.cursor() as cur:
                execute_values(cur, """
                    INSERT INTO video_db
                    (video_id, title, start_time, end_time, url, content, content_embedding)
                    VALUES %s
                """, rows)

            conn.commit()


# ==================== INIT ====================

def init_all():
    print("\n==============================")
    print("DB INIT START")
    print("==============================")

    create_web_db(WEB_FOLDER)
    create_book_db()

    asyncio.run(create_video_db(SRT_FOLDER))

    print("\n==============================")
    print("DONE")
    print("==============================")


if __name__ == "__main__":
    init_all()