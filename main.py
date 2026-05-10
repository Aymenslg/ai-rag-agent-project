from rag import build_index
from workflow import process_query


def main():
    print("=== AI PROJECT (RAG + AGENT) ===\n")

    print("🔄 Loading documents...")
    index = build_index()

    if index is None:
        print("⚠️ No documents found. RAG will be limited.\n")
    else:
        print("✅ Documents loaded successfully!\n")

    print("💡 You can:")
    print("- Ask questions about your documents")
    print("- Use tools: summarize, search, or math (e.g., 2+2)")
    print("- Type 'exit' to quit\n")

    while True:
        query = input("You: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Bye 👋")
            break

        if not query:
            continue

        try:
            response = process_query(index, query)

            print("\n🤖 Answer:")
            print(response)
            print()

        except Exception as e:
            print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    main()