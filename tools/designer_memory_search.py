from langchain_core.tools import tool
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import os

@tool
def designer_memory_search(query: str) -> str:
    """
    Retrieve past design notes or text content from  FAISS vector store.
    """
    embedding_model = OpenAIEmbeddings()
    index_path = "rag/data_store/faiss_text"

    if not os.path.exists(index_path):
        return "No text memory found."

    db = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
    results = db.similarity_search(query, k=3)

    if not results:
        return "No relevant results found in memory."

    return "\n\n".join([doc.page_content for doc in results])
