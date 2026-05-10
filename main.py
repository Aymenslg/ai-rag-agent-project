from rag import build_index
from workflow import process_query


def main():
    print("AI Project")
    print()

    print("Loading documents...")
    index = build_index()

    if index is None:
        print("No documents found. RAG features are limited.")
    else:
        print("Documents loaded successfully.")

    print()
    print("Available features:")
    print("- Ask questions about documents")
    print("- Summarize text")
    print("- Perform calculations")
    print("- Simulate web search")
    print("- Type 'exit' to quit")
    print()

    while True:
        query = input("You: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Goodbye")
            break

        if not query:
            continue

        try:
            response = process_query(index, query)

            print()
            print("Answer:")
            print(response)
            print()

        except Exception as e:
            print(f"Error: {e}")
            print()


if __name__ == "__main__":
    main()