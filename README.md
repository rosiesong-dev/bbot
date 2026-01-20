# 🤖 BeBot: 창조과학 RAG 챗봇 🤖

BeBot은 **LangGraph 기반 PostgreSQL RAG(Retrieval-Augmented Generation) 챗봇**으로, 창조과학 관련 질문에 대해 문서 검색 → 판단 → 생성/재작성 과정을 거쳐 성경적 관점에 기반한 답변을 제공합니다.

---

## 🔹 주요 기능

- 로컬 PostgreSQL DB에 크롤링 데이터 저장
- 텍스트 임베딩을 PostgreSQL `vector` 컬럼에 저장 (pgvector 활용)
- 질문에 맞는 관련 문서 검색 (RAG)
- OpenAI / Upstage API 기반 자연어 생성
- 한국어/영어 자동 감지 및 답변 생성


---

## ⚙️ 설치 및 환경 설정

1. **Python 가상환경 생성**
```bash
conda create -n bbot python=3.12
conda activate bbot
```

2. 필요 패키지 설치
```
pip install -r requirements.txt
```

3. 환경 변수 설정(.env)
```
UPSTAGE_API_KEY=<YOUR_UPSTAGE_API_KEY>
UPSTAGE_BASE_URL=<YOUR_UPSTAGE_BASE_URL>
DB_HOST=localhost
DB_NAME=bbot_db
DB_USER=rosie
DB_PASSWORD=""
DB_PORT=5433
```

4.	PostgreSQL 및 pgvector 설치
```
brew install postgresql@18
brew services start postgresql@18
brew install pgvector
```


## 🧠 LangGraph 기반 질의 처리 아키텍처

BeBot은 LangGraph를 사용하여 질문 처리 흐름을 **그래프 구조로 제어**합니다.  
단순 RAG가 아닌, *답변 적합성 판단과 재검색 루프*를 포함합니다.

### 🔁 Graph Flow

route → retrieve → judge  
        ↓  
      [resolved?]  
     ↙    ↘  
  generate  rewrite  
   ↓     ↓  
   END   retrieve  
       ↓  
       judge


### 📌 Node 설명
---
- **route**
  - 질문 유형 분류 (창조과학 / 성경 / 일반)
  - 처리 전략 결정

- **retrieve**
  - PostgreSQL + pgvector 기반 문서 검색
  - 질문 임베딩과 문서 임베딩 유사도 계산

- **judge**
  - 검색 결과가 질문에 충분히 답변 가능한지 판단
  - 기준: 관련성, 정보 충분성

- **generate**
  - 검색된 문서를 바탕으로 최종 답변 생성
  - 성경적 세계관 + 창조과학 관점 반영

- **rewrite**
  - 질문이 모호하거나 검색 실패 시
  - 질문을 더 명확하게 재작성

- **END**
  - 답변 완료



### 🔍  질의 처리 흐름 요약
---

1. 사용자 질문 입력
2. LangGraph `route` 노드에서 질문 유형 분기
3. `retrieve` 노드에서 관련 문서 검색
4. `judge` 노드에서 답변 가능 여부 판단
5. 
   - 가능 → `generate` → 답변 출력
   - 불충분 → `rewrite` → 재검색 루프

   

## 🚀 실행 방법
1. streamlit UI 실행
```
streamlit run bbot_ui.py
```

2. 웹 브라우저에서 접속 후 질문
</br>
•	사용자 질문 입력 → RAG 기반 답변 생성 </br>
•	한국어/영어 자동 감지


--- 
## 📦 최종 산출물
	•	bbot_ui.py : Streamlit UI 
	•	bbot.py : 핵심 로직 및 DB 연동
	•	bbotCss.py: UI style 	
	•	extracted_texts : 크롤링 데이터
	•	PostgreSQL DB (bbot_db)
	•	.env : 환경 변수
	•	Streamlit 데모 화면에서 생성된 질의 응답 기록

---
## 📌 주의 사항
	•	pgvector 확장 설치 필수 (PostgreSQL 15 기준)
	•	임베딩 차원 수(4096)는 Upstage 모델 기준
	•	로컬 DB 실행 후, create_db() 실행 필요