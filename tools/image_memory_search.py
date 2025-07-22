from langchain_core.tools import tool
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from PIL import Image
import numpy as np
import os
import io

clip_model = SentenceTransformer("clip-ViT-B-32")

def embed_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image_np = np.asarray(image) / 255.0
    return clip_model.encode(image_np, convert_to_tensor=False)

@tool
def image_memory_search(image_bytes: bytes) -> str:
    """
    Retrieve similar past images from local FAISS image vector store.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        return f"Failed to process image: {str(e)}"

    query_vector = embed_image(image)
    index_path = "rag/data_store/faiss_image"

    if not os.path.exists(index_path):
        return "No image memory found."

    db = FAISS.load_local(index_path, clip_model, allow_dangerous_deserialization=True)
    results = db.similarity_search_by_vector(query_vector, k=3)

    if not results:
        return "No similar images found."

    return "\n\n".join([doc.metadata.get("caption", "Unnamed image")] for doc in results)
