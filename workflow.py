from rag import query_index, count_documents
from agent import run_agent


def process_query(index, query: str) -> str:
    query_lower = query.lower().strip()

    try:
        document_queries = [
            "how many document",
            "how many documents",
            "number of documents",
            "count documents",
            "files do i have"
        ]

        if any(text in query_lower for text in document_queries):
            count, names = count_documents()

            if count == 0:
                return "No documents found"

            return f"You have {count} documents: {', '.join(names)}"

        if index is not None:
            rag_response = query_index(index, query)

            if (
                rag_response
                and "no documents" not in rag_response.lower()
            ):
                return rag_response

        return run_agent(query)

    except Exception as e:
        return f"Workflow error: {e}"