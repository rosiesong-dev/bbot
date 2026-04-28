import argparse
import time
from datetime import timedelta
from bbot_graph import generate


def format_timedelta(td: timedelta) -> str:
    total = int(td.total_seconds())
    h, r = divmod(total, 3600)
    m, s = divmod(r, 60)
    return f"{h:02}:{m:02}:{s:02}"


def handle_question(question):

    try:
        start_time = time.time()

        # =========================
        # 1. Generate Answer
        # =========================
        answer, sources_info = generate(question)
        elapsed = time.time() - start_time

        print("\n" + "=" * 60)
        print("💬 Answer")
        print("=" * 60)
        print(answer)

        # =========================
        # 2. Context 정리 (RAGAS용)
        # =========================
        contexts = []

        if sources_info.get("book_docs"):
            contexts.extend(
                [
                    doc["content"]
                    for doc in sources_info["book_docs"]
                    if doc.get("content")
                ]
            )

        if sources_info.get("web_docs"):
            contexts.extend(
                [
                    doc["content"]
                    for doc in sources_info["web_docs"]
                    if doc.get("content")
                ]
            )

        if not contexts:
            contexts = ["No context available"]

        # =========================
        # 5. Sources 출력
        # =========================
        sources = []

        if sources_info.get("video_docs"):
            sources.append("\n🎬 Video Sources:")
            for doc in sources_info["video_docs"]:
                start = format_timedelta(timedelta(seconds=int(doc["start"])))
                end = format_timedelta(timedelta(seconds=int(doc["end"])))
                sources.append(f"  - [{doc['title']}] {doc['url']} ({start}-{end})")

        if sources_info.get("web_docs"):
            web_urls = list(
                set(d["url"] for d in sources_info["web_docs"] if d.get("url"))
            )

            if web_urls:
                sources.append("\n🌐 Web Sources:")
                for url in web_urls:
                    sources.append(f"  - {url}")

        if sources_info.get("book_docs"):
            sources.append("\n📖 Book Sources:")
            for doc in sources_info["book_docs"]:
                sources.append(f"  - {doc['book']} (p{doc['page']})")

        if sources:
            print("\n" + "-" * 60)
            print("📚 Sources")
            print("-" * 60)
            print("\n".join(sources))

        print("-" * 60)
        print(f"⏱️ Duration: {elapsed:.2f} seconds")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ Error occurred: {e}\n")


def ask_user():

    print("\n" + "=" * 60)
    print("  BeBot CLI — 'exit' or Ctrl+C to quit")
    print("=" * 60 + "\n")

    while True:

        try:
            question = input("❓ Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Terminating...")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit", "종료"):
            print("👋 Terminating...")
            break

        handle_question(question)


def parse_args():

    parser = argparse.ArgumentParser(description="BeBot CLI")
    parser.add_argument("--question", type=str, help="질문을 입력하세요.")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    if args.question:
        handle_question(args.question)
    else:
        ask_user()
