from pathlib import Path

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings
)

from llama_index.embeddings.huggingface import (
    HuggingFaceEmbedding
)

from llama_index.llms.ollama import Ollama


def build_index():
    try:
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        Settings.llm = Ollama(
            model="qwen2.5:1.5b",
            request_timeout=120.0
        )

        data_path = Path("./data")

        if not data_path.exists():
            return None

        documents = SimpleDirectoryReader("./data").load_data()

        if not documents:
            return None

        return VectorStoreIndex.from_documents(documents)

    except Exception as e:
        print(f"RAG build error: {e}")
        return None


def query_index(index, question):
    if index is None:
        return "No documents available"

    try:
        query_engine = index.as_query_engine(
            similarity_top_k=3
        )

        response = query_engine.query(question)

        return str(response)

    except Exception as e:
        return f"Query error: {e}"


def count_documents():
    files = list(Path("./data").glob("*.txt"))

    if not files:
        return 0, []

    return len(files), [file.name for file in files]