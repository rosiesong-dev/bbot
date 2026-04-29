import os
from dotenv import load_dotenv

load_dotenv()

# ==================== Provider ====================
PROVIDER = os.getenv("PROVIDER", "gemma") # "upstage", "gemma", "openai" 


# ==================== Upstage ====================
UPSTAGE_API_KEY  = os.getenv("UPSTAGE_API_KEY")
UPSTAGE_BASE_URL = os.getenv("UPSTAGE_BASE_URL", "https://api.upstage.ai/v1/solar")
UPSTAGE_LLM_MODEL   = os.getenv("UPSTAGE_LLM_MODEL",   "solar-pro3")
UPSTAGE_EMBED_MODEL = os.getenv("UPSTAGE_EMBED_MODEL",  "embedding-query")
UPSTAGE_EMBED_DIM   = 4096

# ==================== OpenAI ====================
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
OPENAI_LLM_MODEL   = os.getenv("OPENAI_LLM_MODEL",   "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL",  "text-embedding-3-small")
OPENAI_EMBED_DIM   = 1536

# ==================== Gemma ====================
GEMMA_API_KEY = os.getenv("GEMMA_API_KEY")
GEMMA_BASE_URL = os.getenv("GEMMA_BASE_URL")
GEMMA_LLM_MODEL = os.getenv("GEMMA_LLM_MODEL", "gemma-4")
GEMMA_EMBED_MODEL = os.getenv("GEMMA_EMBED_MODEL","text-embedding-3-small")
GEMMA_EMBED_DIM = 2560


# ==================== 현재 Provider 기준 값 ====================
if PROVIDER == "upstage":
    EMBED_DIM = UPSTAGE_EMBED_DIM
    LLM_MODEL = UPSTAGE_LLM_MODEL
elif PROVIDER == "gemma":
    EMBED_DIM = GEMMA_EMBED_DIM
    LLM_MODEL = GEMMA_LLM_MODEL
else:
    EMBED_DIM = OPENAI_EMBED_DIM
    LLM_MODEL = OPENAI_LLM_MODEL


# ==================== DB 접속 정보 ====================
DB_HOST     = os.getenv("DB_HOST")
DB_NAME     = os.getenv("DB_NAME")
DB_USER     = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT     = os.getenv("DB_PORT")

# ==================== DB 연결 함수 ====================
def get_conn():
    import psycopg2
    return psycopg2.connect(
        host=DB_HOST, dbname=DB_NAME, user=DB_USER,
        password=DB_PASSWORD, port=DB_PORT
    )