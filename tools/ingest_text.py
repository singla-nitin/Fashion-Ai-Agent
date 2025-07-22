from langchain_core.tools import tool
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import os

@tool
def ingest_design_text(path: str) -> str:
    """
    Ingests all .txt files in `path` into your FAISS text index.
    """
    embedding_model = OpenAIEmbeddings()
    docs = []
    # Load and chunk
    for fname in os.listdir(path):
        if fname.endswith(".txt"):
            loader = TextLoader(os.path.join(path, fname))
            docs.extend(loader.load())
    if not docs:
        return "No .txt files found to ingest."

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Build or overwrite FAISS index
    index_path = "rag/data_store/faiss_text"
    db = FAISS.from_documents(chunks, embedding_model)
    os.makedirs(index_path, exist_ok=True)
    db.save_local(index_path)
    return f"Ingested {len(chunks)} text chunks into FAISS."
