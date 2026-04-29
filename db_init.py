# =========================
# DB INIT (STABLE VERSION)
# =========================

import os
import asyncio
import pdfplumber
import pandas as pd
import unicodedata
from datetime import datetime
from psycopg2.extras import execute_values

from config import get_conn, EMBED_DIM, PROVIDER
from llm_factory import get_embedding, get_llm

embedding_model = get_embedding()
llm_model = get_llm()


# =========================
# EMBEDDING WRAPPER (핵심)
# =========================

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


def get_batch_embeddings(texts: list[str]):
    texts = [safe_text(t) for t in texts if t]
    texts = [t for t in texts if t]

    if not texts:
        return []

    if PROVIDER == "gemma":
        return embedding_model.encode(texts).tolist()

    return embedding_model.embed_documents(texts)


# =========================
# TABLE CHECK
# =========================

def table_exists(table_name: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = %s
                );
            """, (table_name,))
            return cur.fetchone()[0]


# =========================
# IMAGE TABLE
# =========================

def create_image_table():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS book_images (
                    id SERIAL PRIMARY KEY,
                    book_name TEXT,
                    page_num INT,
                    file_name TEXT,
                    file_path TEXT
                );
            """)
            conn.commit()

    print("🖼️ image table ready")


# =========================
# BOOK DB
# =========================

BOOKS_FOLDER = "./books"


def create_book_db():
    print("\n📚 BOOK DB START")

    for lang in ["en", "ko"]:
        path = os.path.join(BOOKS_FOLDER, lang)

        if not os.path.exists(path):
            print(f"skip {path}")
            continue

        table = f"book_{lang}"

        with get_conn() as conn:
            with conn.cursor() as cur:
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
            pdf_path = os.path.join(path, pdf)
            book_name = pdf.replace(".pdf", "")

            try:
                with pdfplumber.open(pdf_path) as pdf_obj:
                    with get_conn() as conn:
                        with conn.cursor() as cur:

                            for i, page in enumerate(pdf_obj.pages, 1):
                                text = page.extract_text()

                                if not text or len(text.strip()) < 50:
                                    continue

                                vec = get_text_embedding(text)

                                cur.execute(f"""
                                    INSERT INTO {table}
                                    (book_name, page_num, content, embedding)
                                    VALUES (%s, %s, %s, %s::vector)
                                """, (book_name, i, text, vec))

                            conn.commit()

                print(f"✔ {book_name} done")

            except Exception as e:
                print(f"❌ {book_name} error: {e}")


# =========================
# WEB DB
# =========================

WEB_FOLDER = "./extracted_texts"


def create_web_db():
    print("\n📁 WEB DB START")

    if not os.path.exists(WEB_FOLDER):
        print("no web folder")
        return

    files = [f for f in os.listdir(WEB_FOLDER) if f.endswith(".txt")]

    with get_conn() as conn:
        with conn.cursor() as cur:

            for f in files:
                try:
                    with open(os.path.join(WEB_FOLDER, f), "r", encoding="utf-8") as fp:
                        text = fp.read()

                    vec = get_text_embedding(text)

                    cur.execute("""
                        INSERT INTO crawled_data (title, content, content_embedding)
                        VALUES (%s, %s, %s::vector)
                    """, (f, text, vec))

                except Exception as e:
                    print(f"❌ {f} failed: {e}")

            conn.commit()


# =========================
# INIT ALL
# =========================

def init_all():
    print("\n==============================")
    print("DB INIT START")
    print("==============================\n")

    create_image_table()

    # web
    if not table_exists("crawled_data"):
        create_web_db()

    # book
    create_book_db()

    print("\n==============================")
    print("DB INIT DONE")
    print("==============================\n")


if __name__ == "__main__":
    init_all()