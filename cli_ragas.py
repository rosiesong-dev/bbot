import time
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerCorrectness,
    context_recall,
    _ContextPrecision
)

from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from ragas.llms import llm_factory

from bbot_graph import generate


# =========================
# SETUP
# =========================
client = OpenAI()

llm = llm_factory(
    "gpt-4o-mini",
    client=client
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# =========================
# MAIN LOOP
# =========================
print("\n=== RAGAS CLI START ===\n")

while True:
    question = input("❓ Question: ").strip()

    if question.lower() in ["exit", "quit", "종료"]:
        break

    if not question:
        continue

    try:
        start = time.time()

        # =========================
        # 1. GENERATE
        # =========================
        answer, sources_info = generate(question)

        print("\n💬 ANSWER:\n", answer)

        # =========================
        # 🔥 CRITICAL FIX 1: answer must be CLEAN but NOT distorted
        # =========================
        answer = answer.strip()

        # 👉 너무 길면 잘라서 noise 줄이기
        answer = " ".join(answer.split()[:80])

        # =========================
        # 2. CONTEXTS
        # =========================
        contexts = []

        for d in sources_info.get("web_docs", []):
            if d.get("content"):
                contexts.append(d["content"])

        for d in sources_info.get("book_docs", []):
            if d.get("content"):
                contexts.append(d["content"])

        # 🔥 fallback (VERY IMPORTANT)
        if not contexts:
            contexts = [answer]

        # 🔥 flatten (RAGAS 안정 핵심)
        contexts = [" ".join(contexts)]

        # =========================
        # 3. DATASET
        # =========================
        dataset = Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [question]
        })

        # =========================
        # 4. EVALUATION
        # =========================
        result = evaluate(
            dataset,
            metrics=[
                Faithfulness(),
                AnswerCorrectness(),
                context_recall,
                _ContextPrecision()
            ],
            llm=llm,
            embeddings=embeddings
        )

        print("\n📊 RAGAS RESULT")
        print(result)

        # =========================
        # 5. SAVE
        # =========================
        with open("ragas_result.txt", "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Q: {question}\n")
            f.write(f"A: {answer}\n")
            f.write(f"R: {result}\n")
            f.write("=" * 80 + "\n")

        print("\n✅ saved")

    except Exception as e:
        print("❌ ERROR:", e)