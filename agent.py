from langchain_ollama import ChatOllama
from tools import calculator, summarize, web_search


llm = ChatOllama(model="qwen2.5:1.5b")


def is_math_expression(text: str) -> bool:
    """Check if text is a valid math expression."""
    allowed_chars = "0123456789+-*/(). "
    return all(c in allowed_chars for c in text) and any(op in text for op in "+-*/")


def run_agent(query: str):
    query_clean = query.strip()
    query_lower = query_clean.lower()

    try:
        # 🔢 CALCUL (sécurisé)
        if is_math_expression(query_clean):
            return calculator.invoke({"expression": query_clean})

        # 📄 SUMMARIZE
        if query_lower.startswith("summarize"):
            text = query_clean[len("summarize"):].strip()

            if not text:
                return "Please provide text after 'summarize'."

            return summarize.invoke({"text": text})

        # 🌐 WEB SEARCH
        if query_lower.startswith("search"):
            text = query_clean[len("search"):].strip()

            if not text:
                return "Please provide a query after 'search'."

            return web_search.invoke({"query": text})

        # 🤖 LLM (fallback)
        response = llm.invoke(query_clean)
        return response.content

    except Exception as e:
        return f"Agent error: {e}"