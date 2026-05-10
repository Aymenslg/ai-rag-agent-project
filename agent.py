from langchain_ollama import ChatOllama
from tools import calculator, summarize, web_search


llm = ChatOllama(model="qwen2.5:1.5b")


def is_math_expression(text: str) -> bool:
    allowed_chars = "0123456789+-*/(). "
    return all(c in allowed_chars for c in text) and any(
        op in text for op in "+-*/"
    )


def run_agent(query: str):
    query = query.strip()
    query_lower = query.lower()

    try:
        if is_math_expression(query):
            return calculator.invoke({"expression": query})

        if query_lower.startswith("summarize"):
            text = query[len("summarize"):].strip()

            if not text:
                return "Please provide text to summarize."

            return summarize.invoke({"text": text})

        if query_lower.startswith("search"):
            text = query[len("search"):].strip()

            if not text:
                return "Please provide a search query."

            return web_search.invoke({"query": text})

        response = llm.invoke(query)
        return response.content

    except Exception as e:
        return f"Agent error: {e}"