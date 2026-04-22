# server.py — FastAPI 서버 (Flutter 앱 ↔ Python 백엔드 브릿지)
#
# 기존 bbot_graph.generate() 함수를 REST API로 노출합니다.
# 실행: uvicorn server:app --host 0.0.0.0 --port 8000
#
# 엔드포인트:
#   POST /api/chat      — 질문 → 답변 + 출처
#   GET  /api/health     — 서버 상태 확인

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="BeBot API", version="1.0.0")

# CORS 설정 (Flutter 앱에서 접근 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────── Request / Response 모델 ────────
class ChatRequest(BaseModel):
    question: str


class SourceDoc(BaseModel):
    title: Optional[str] = ""
    url: Optional[str] = ""
    content: Optional[str] = ""
    book: Optional[str] = ""
    page: Optional[int] = 0
    video_id: Optional[str] = ""
    start: Optional[float] = 0
    end: Optional[float] = 0


class ChatResponse(BaseModel):
    answer: str
    sources: dict          # { web_docs: [...], book_docs: [...], video_docs: [...] }
    top_sources: list      # 관련성 상위 3개 (score 기준, UI 표시용)


# ──────── 엔드포인트 ────────
@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "BeBot API"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    사용자 질문을 받아 bbot_graph.generate()를 호출하고
    답변 + 출처 정보를 JSON으로 반환합니다.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="질문이 비어있습니다.")

    try:
        from bbot_graph import generate

        answer, sources_raw = generate(req.question.strip())

        # 출처 데이터 정리
        raw = sources_raw if isinstance(sources_raw, dict) else {}
        sources = {
            "web_docs":   raw.get("web_docs", []),
            "book_docs":  raw.get("book_docs", []),
            "video_docs": raw.get("video_docs", []),
        }

        # 전체 출처를 score 기준 정렬 후 상위 3개 선별 (UI 표시용)
        all_docs = sources["web_docs"] + sources["book_docs"] + sources["video_docs"]
        top_sources = []
        for doc in sorted(all_docs, key=lambda d: d.get("score", 0))[:3]:
            t = doc.get("type")
            if t == "video":
                top_sources.append({"type": t, "title": doc.get("title", ""), "url": doc.get("url", ""), "start": doc.get("start", 0), "end": doc.get("end", 0)})
            elif t == "book":
                top_sources.append({"type": t, "title": doc.get("book", ""), "page": doc.get("page", 0), "content": doc.get("content", "")})
            else:
                top_sources.append({"type": t, "title": doc.get("title", ""), "url": doc.get("url", ""), "content": doc.get("content", "")})

        return ChatResponse(answer=answer, sources=sources, top_sources=top_sources)

    except Exception as e:
        print(f"❌ 답변 생성 오류: {e}")
        raise HTTPException(status_code=500, detail=f"답변 생성 중 오류 발생: {str(e)}")


# ──────── 직접 실행 시 ────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)