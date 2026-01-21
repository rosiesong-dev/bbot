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

model = ChatUpstage(api_key=api_key, base_url=base_url)
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
    category: str = Field(description="The category of the text segment. Options: 'Intro', 'Outro', 'Content', 'Mixed', 'Noise'")
    reasoning: str = Field(description="Brief reasoning why this segment belongs to the category.")
    is_useful: bool = Field(description="Set to True if the text contains core content. False if purely intro/outro/noise.")
    cleaned_text: str = Field(description="The cleaned English text with noise removed.")


# ==================== SRT 프로세서 ====================
class SrtProcessor:
    def __init__(self, embedding_model, llm_model):
        self.embedding_model = embedding_model
        self.llm = llm_model
        
        # 의미 기반 청커
        self.semantic_chunker = SemanticChunker(
            embeddings=embedding_model,
            breakpoint_threshold_type="gradient", 
        )
        
        # 강제 분할용 청커
        self.force_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
            separators=[".", " ", ""]
        )

    def _parse_srt(self, file_path: str) -> List[Dict]:
        """SRT 파싱"""
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

    def _semantic_chunking_with_safety_merge(self, srt_items: List[Dict]) -> List[Dict]:
        """Hybrid Chunking: Semantic + Length Check"""
        print("✂️ Semantic chunking...")
        sentences = [item["text"] for item in srt_items]
        if not sentences:
            return []
        
        raw_chunks = self.semantic_chunker.split_text("\n".join(sentences))
        
        aligned_chunks = []
        cursor = 0
        
        for chunk_text in raw_chunks:
            chunk_sentences = [s.strip() for s in chunk_text.split("\n") if s.strip()]
            if not chunk_sentences: 
                continue
            
            chunk_len = len(chunk_sentences)
            
            start_idx = min(cursor, len(srt_items)-1)
            end_idx = min(cursor + chunk_len - 1, len(srt_items)-1)
            
            start_item = srt_items[start_idx]
            end_item = srt_items[end_idx]
            
            primary_chunk = {
                "start_time": start_item["start"],
                "end_time": end_item["end"],
                "content": " ".join(chunk_sentences)
            }
            
            # Hard Limit Check
            if len(primary_chunk["content"]) > 1200:
                split_sub_chunks = self._force_split_chunk(primary_chunk)
                aligned_chunks.extend(split_sub_chunks)
            else:
                aligned_chunks.append(primary_chunk)

            cursor += chunk_len

        return aligned_chunks

    def _force_split_chunk(self, huge_chunk: Dict) -> List[Dict]:
        """너무 긴 청크 강제 분할"""
        full_text = huge_chunk["content"]
        start_t = huge_chunk["start_time"]
        end_t = huge_chunk["end_time"]
        total_duration = end_t - start_t
        total_len = len(full_text)
        
        sub_texts = self.force_splitter.split_text(full_text)
        
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

    async def _refine_chunk_async(self, chunk: Dict, chain) -> Optional[Dict]:
        """단일 청크 비동기 필터링"""
        try:
            res = await chain.ainvoke({"text": chunk["content"]})
            if res and res.is_useful and res.cleaned_text.strip():
                if len(res.cleaned_text) < 10: 
                    return None
                chunk["content"] = res.cleaned_text
                return chunk
        except Exception as e:
            print(f"Processing error: {e}")
            pass
        return None

    async def process_file_async(self, file_path: str) -> List[Dict]:
        """전체 파이프라인 (Async)"""
        srt_items = self._parse_srt(file_path)
        
        raw_chunks = self._semantic_chunking_with_safety_merge(srt_items)
        
        if not raw_chunks:
            return []

        print(f"🧹 Refining {len(raw_chunks)} chunks in parallel...")

        system_prompt = """
        You are an expert subtitle editor.
        Your goal is to transform raw spoken text into clean, readable prose.
        
        [RULES]
        1. **Analyze**: Check if the text is Intro, Outro, or Content.
        2. **Remove Noise**: Remove fillers (um, uh, like), greetings, and self-introductions.
        3. **Fix Structure**: The input text is a run-on sentence. **You MUST split it into proper sentences with punctuation.**
        4. **Keep Facts**: Do not summarize details.
        
        Input: "hello my name is matt uh the ark was big it was 300 cubits"
        Output Cleaned: "The ark was big. It was 300 cubits."
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Input Text: {text}")
        ])

        chain = prompt | self.llm.with_structured_output(FilteredContent)

        tasks = [self._refine_chunk_async(chunk, chain) for chunk in raw_chunks]
        refined_results = await asyncio.gather(*tasks)

        final_chunks = [r for r in refined_results if r is not None]
        print(f"✅ Final valid chunks: {len(final_chunks)}")
        return final_chunks


# ==================== DB Repository ====================
class VideoRepository:
    def init_db(self):
        """테이블 생성"""
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                create_table_sql = """
                CREATE TABLE IF NOT EXISTS video_segments (
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
                """
                cur.execute(create_table_sql)
                conn.commit()
                print("[Video DB] 초기화 완료\n")

    def delete_video_data(self, video_id: str):
        """기존 데이터 삭제"""
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM video_segments WHERE video_id = %s", (video_id,))
                conn.commit()
                print(f"[Video DB] '{video_id}' 기존 데이터 삭제")

    def save_batch(self, data_tuples: List[Tuple]):
        """데이터 일괄 삽입"""
        with get_conn() as conn:
            with conn.cursor() as cur:
                insert_sql = """
                INSERT INTO video_segments 
                (video_id, title, start_time, end_time, url, content, content_embedding)
                VALUES %s
                """
                
                execute_values(cur, insert_sql, data_tuples)
                conn.commit()
                print(f"[Video DB] {len(data_tuples)}개 청크 삽입 완료\n")


# ==================== 메인 파이프라인 ====================
async def main_pipeline(srt_path: str, video_meta: dict):   
    repo = VideoRepository()
    processor = SrtProcessor(embedding_model, model)

    repo.init_db()

    if not os.path.exists(srt_path):
        print(f"File not found: {srt_path}")
        return

    print(f"🔄 Processing [{video_meta['video_id']}] {video_meta['title']}...")
    
    # 파싱 -> 청킹 -> LLM 정제
    refined_chunks = await processor.process_file_async(srt_path)
    
    if not refined_chunks:
        print("No valid content found. Skipping.\n")
        return

    # 임베딩 생성
    print(f"🔮 Generating embeddings for {len(refined_chunks)} chunks...")
    texts = [c["content"] for c in refined_chunks]
    vectors = embedding_model.embed_documents(texts)

    # 기존 데이터 삭제
    repo.delete_video_data(video_meta["video_id"])

    # DB 저장 준비
    data_tuples = []
    for i, chunk in enumerate(refined_chunks):
        time_url = f"{video_meta['url']}?t={int(chunk['start_time'])}s"
        
        row = (
            video_meta["video_id"],
            video_meta["title"],
            chunk["start_time"],
            chunk["end_time"],
            time_url,
            chunk["content"],
            vectors[i]
        )
        data_tuples.append(row)

    # 실제 저장
    repo.save_batch(data_tuples)


# ==================== 폴더 일괄 처리 ====================
async def process_directory(directory_path: str):
    """폴더 내 모든 SRT 파일 처리"""
    if not os.path.exists(directory_path):
        print(f"❌ Error: Directory '{directory_path}' not found.")
        return

    files = sorted([f for f in os.listdir(directory_path) if f.endswith(".srt")])
    
    if not files:
        print("⚠️ No .srt files found.")
        return

    print(f"📦 Found {len(files)} SRT files.\n")

    for idx, filename in enumerate(files, start=1):
        file_path = os.path.join(directory_path, filename)
        
        # 파일명 파싱
        base_name = filename.rsplit(".srt", 1)[0]
        
        try:
            if "_" in base_name:
                title_part, id_part = base_name.rsplit("_", 1)
                youtube_id = id_part.split(".")[0]
            else:
                title_part = base_name
                youtube_id = ""
                print(f"⚠️ Warning: Cannot parse ID from '{filename}'.")

        except Exception as e:
            print(f"Error parsing filename '{filename}': {e}")
            title_part = base_name
            youtube_id = ""

        # 메타데이터 생성
        db_video_id = f"v{idx:04d}"
        title = title_part.strip()
        url = f"https://youtu.be/{youtube_id}" if youtube_id else ""

        meta = {
            "video_id": db_video_id,
            "title": title,
            "url": url
        }

        # 파이프라인 실행
        await main_pipeline(file_path, meta)

    print("="*60)
    print("✅ All videos processed!")
    print("="*60)



# ==================== 벡터 검색 ====================
def retrieve_video_segments(question: str, top_k: int = 5):
    print(f"\n🔎 질문: {question}")

    # 1️⃣ 질문 임베딩
    q_emb = embedding_model.embed_query(question)
    print("🧠 질문 임베딩 생성 완료")

    # 2️⃣ 벡터 검색
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    video_id,
                    title,
                    start_time,
                    end_time,
                    url,
                    content
                FROM video_segments
                ORDER BY content_embedding <#> %s::vector
                LIMIT %s
            """, (q_emb, top_k))

            rows = cur.fetchall()

    print(f"📄 검색 결과 수: {len(rows)}")

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
    SRT_DIRECTORY = "./srt_data"
    
    if not os.path.exists(SRT_DIRECTORY):
        print(f"'{SRT_DIRECTORY}' 폴더에 .srt 파일을 넣어주세요.")
    else:
        asyncio.run(process_directory(SRT_DIRECTORY))