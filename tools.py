from langchain_core.tools import tool
from langchain_ollama import ChatOllama
import math

# LLM local
llm = ChatOllama(model="qwen2.5:1.5b")


# 🔢 CALCULATEUR (sécurisé)
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

        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)

    except Exception:
        return "Error in calculation"


# 📄 SUMMARIZE (amélioré)
@tool
def summarize(text: str) -> str:
    """Summarize a text using a local LLM."""

    text = text.strip()

    if not text:
        return "Error: no text provided."

    # 🔒 limite pour performance
    text = text[:1000]

    prompt = f"""
    Give a concise summary (1-2 sentences max) of the following text:

    {text}
    """

    try:
        response = llm.invoke(prompt)
        return response.content.strip()

    except Exception as e:
        return f"Summarization error: {e}"


# 🌐 WEB SEARCH (offline)
@tool
def web_search(query: str) -> str:
    """Simulate a web search (offline mode)."""
    return f"[Offline mode] No internet access. General answer: {query}"