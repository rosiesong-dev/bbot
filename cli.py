import time
from datetime import timedelta
from bbot_graph import generate


def format_timedelta(td: timedelta) -> str:
    total = int(td.total_seconds())
    h, r  = divmod(total, 3600)
    m, s  = divmod(r, 60)
    return f"{h:02}:{m:02}:{s:02}"


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  BeBot CLI — 종료하려면 'exit' 또는 Ctrl+C")
    print("="*60 + "\n")

    while True:
        try:
            question = input("❓ 질문: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 종료합니다.")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "종료"):
            print("👋 종료합니다.")
            break

        try:
            start_time = time.time()
            answer, sources_info = generate(question)
            elapsed = time.time() - start_time

            print("\n" + "="*60)
            print("💬 답변")
            print("="*60)
            print(answer)

            sources = []
            if sources_info.get("video_docs"):
                sources.append("\n🎬 영상 자료:")
                for doc in sources_info["video_docs"]:
                    start = format_timedelta(timedelta(seconds=int(doc["start"])))
                    end   = format_timedelta(timedelta(seconds=int(doc["end"])))
                    sources.append(f"  - [{doc['title']}] {doc['url']} ({start}-{end})")

            if sources_info.get("web_docs"):
                web_urls = list(set(d["url"] for d in sources_info["web_docs"] if d.get("url")))
                if web_urls:
                    sources.append("\n🌐 웹 자료:")
                    for url in web_urls:
                        sources.append(f"  - {url}")

            if sources_info.get("book_docs"):
                sources.append("\n📖 책 자료:")
                for doc in sources_info["book_docs"]:
                    sources.append(f"  - {doc['book']} (p{doc['page']})")

            if sources:
                print("\n" + "-"*60)
                print("📚 출처")
                print("-"*60)
                print("\n".join(sources))

            print("-"*60)
            print(f"⏱️  소요 시간: {elapsed:.2f}초")
            print("="*60 + "\n")

        except Exception as e:
            print(f"\n❌ 오류 발생: {e}\n")