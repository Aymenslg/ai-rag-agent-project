from langchain_core.tools import tool
from langchain_ollama import ChatOllama
import math


llm = ChatOllama(model="qwen2.5:1.5b")


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely."""

    try:
        allowed_names = {
            "abs": abs,
            "round": round,
            "sqrt": math.sqrt,
            "pow": pow
        }

        result = eval(
            expression,
            {"__builtins__": {}},
            allowed_names
        )

        return str(result)

    except Exception:
        return "Calculation error"


@tool
def summarize(text: str) -> str:
    """Generate a short summary using the local model."""

    text = text.strip()

    if not text:
        return "No text provided"

    text = text[:1000]

    prompt = f"""
    Give a concise summary of the following text in one or two sentences:

    {text}
    """

    try:
        response = llm.invoke(prompt)
        return response.content.strip()

    except Exception as e:
        return f"Summarization error: {e}"


@tool
def web_search(query: str) -> str:
    """Simulate a web search in offline mode."""

    return f"No internet access. General information about: {query}"