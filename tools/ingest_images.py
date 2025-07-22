from langchain_core.tools import tool
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from PIL import Image
import numpy as np
import os
import io

clip_model = SentenceTransformer("clip-ViT-B-32")

def embed_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((224, 224))
    arr = np.asarray(image) / 255.0
    return clip_model.encode(arr, convert_to_tensor=False)
    

@tool
def ingest_design_images(image_path: str, caption: str) -> str:
    """
    Saves a single image and its caption to the long-term FAISS image memory. 
    Use this tool when you feel this image to be an important refrence for designer's mindset.
    """
    try:
        if not os.path.exists(image_path):
            return f"Error: Image path does not exist: {image_path}"

        # 1. Open and embed the single image
        img = Image.open(image_path)
        vector = embed_image(img) # Your existing embed_image function
        
        # The FAISS library expects lists of embeddings and metadata
        vectors = [vector]
        metadatas = [{"caption": caption}]

        # 2. Load existing index or create a new one
        index_path = "rag/data_store/faiss_image"
        os.makedirs(index_path, exist_ok=True)
        
        if os.path.exists(os.path.join(index_path, "index.faiss")):
            db = FAISS.load_local(index_path, clip_model, allow_dangerous_deserialization=True)
            db.add_embeddings(vectors, metadatas)
        else:
            db = FAISS.from_embeddings(vectors, metadatas, clip_model)
        
        # 3. Save the updated index
        db.save_local(index_path)
        
        return f"Successfully added image '{os.path.basename(image_path)}' to memory with caption: '{caption}'."

    except Exception as e:
        return f"An error occurred while adding the image to memory: {e}"

@tool
def ingest_design_images(folder: str) -> str:
    """
    Ingests all images in `folder` into your FAISS image index.
    Filenames should be <name>.<ext>; caption = filename (without ext).
    """
    vectors, metadatas = [], []
    for fname in os.listdir(folder):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(folder, fname)
            img = Image.open(path)
            vectors.append(embed_image(img))
            metadatas.append({"caption": os.path.splitext(fname)[0]})

    if not vectors:
        return "No images found to ingest."

    index_path = "rag/data_store/faiss_image"
    os.makedirs(index_path, exist_ok=True)
    db = FAISS.from_embeddings(vectors, metadatas, clip_model)
    db.save_local(index_path)
    return f"Ingested {len(vectors)} images into FAISS."
