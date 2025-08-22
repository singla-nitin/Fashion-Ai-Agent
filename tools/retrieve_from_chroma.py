import os
from dotenv import load_dotenv
from langchain.tools import tool
from PIL import Image
import io
import torch
import open_clip
from typing import List, Dict, Any

# Disable ChromaDB telemetry to prevent errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Local vector store (ChromaDB)
try:
    from chromadb import PersistentClient  # type: ignore
    _CHROMA_MODE = "persistent"
except Exception:  # pragma: no cover - fallback for older versions
    _CHROMA_MODE = "legacy"
    try:
        import chromadb  # type: ignore
        from chromadb.config import Settings  # type: ignore
    except Exception:
        chromadb = None  # type: ignore
        Settings = None  # type: ignore

load_dotenv()

# Chroma config
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHROMA_DIR = os.path.join(BASE_DIR, "db", "chroma")
os.makedirs(CHROMA_DIR, exist_ok=True)

COLLECTION_NAME = "designer-multimodal-index"

def _get_chroma_collection():
    if _CHROMA_MODE == "persistent":
        try:
            client = PersistentClient(path=CHROMA_DIR)  # type: ignore
            col = client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            return col
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Chroma PersistentClient: {e}")
    else:
        if chromadb is None or Settings is None:
            raise RuntimeError("ChromaDB is not installed. Please install 'chromadb'.")
        try:
            client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DIR))  # type: ignore
            col = client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            return col
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Chroma legacy client: {e}")

device = "cuda" if torch.cuda.is_available() else "cpu"
_clip_bundle = {"model": None, "preprocess": None}
_text_tokenizer = None

def _get_clip():
    if _clip_bundle["model"] is None:
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_e16")
        _clip_bundle["model"] = model.to(device)
        _clip_bundle["preprocess"] = preprocess
    return _clip_bundle["model"], _clip_bundle["preprocess"]

def _get_tokenizer():
    global _text_tokenizer
    if _text_tokenizer is None:
        _text_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return _text_tokenizer

@tool("retrieve_from_chroma")
def retrieve_from_chroma(user_id: str, text: str = None, image_bytes: bytes = None, top_k: int = 5) -> dict:
    """
    Retrieve stored user information from previous sessions (names, preferences, style history) that is NOT available in current conversation memory.
    IMPORTANT: Only use this tool if the information is not already available in your conversation memory from the current chat.
    
    Use this to:
    - Find user preferences from past sessions when they're not in current memory
    - Look up stored user information like names, styles, favorites from previous conversations
    - Search through historical design interactions and preferences
    
    Returns stored user data in JSON format with image_results and text_results arrays.
    """
    results = {"image_results": [], "text_results": []}

    try:
        collection = _get_chroma_collection()
    except Exception as e:
        return {"error": f"ChromaDB is not available: {e}"}

    # Image query
    if image_bytes:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            clip_model, preprocess = _get_clip()
            image_input = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = clip_model.encode_image(image_input)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                embedding = embedding.cpu().numpy().tolist()[0]
        except Exception as e:
            results["image_results"] = [{"error": f"Failed to get CLIP embedding for image: {e}"}]
        else:
            try:
                q = collection.query(
                    query_embeddings=[embedding],
                    n_results=top_k,
                    where={"$and": [
                        {"user_id": {"$eq": user_id}},
                        {"type": {"$eq": "image"}}
                    ]},
                )
                # Chroma returns dict with ids, distances, metadatas
                ids = q.get("ids", [[]])[0]
                distances = q.get("distances", [[]])[0] or q.get("embeddings", [])  # distances may be None depending on version
                metadatas = q.get("metadatas", [[]])[0]
                # Convert distance to similarity (cosine sim ~ 1 - distance) if distances provided
                out = []
                for i, m in enumerate(ids):
                    meta = metadatas[i] if i < len(metadatas) else {}
                    score = None
                    if distances and i < len(distances) and isinstance(distances[i], (int, float)):
                        score = 1 - distances[i]
                    out.append({
                        "id": m,
                        "score": score if score is not None else 0.0,
                        "metadata": meta,
                        "image_data": (meta or {}).get("image_data"),
                    })
                results["image_results"] = out
            except Exception as e:
                results["image_results"] = [{"error": f"Error querying Chroma for image: {e}"}]

    # Text query
    if text:
        try:
            clip_model, _ = _get_clip()
            tokenizer = _get_tokenizer()
            tokens = tokenizer([text])
            with torch.no_grad():
                txt = tokens.to(device)
                embedding = clip_model.encode_text(txt)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                embedding = embedding.cpu().numpy().tolist()[0]
        except Exception as e:
            results["text_results"] = [{"error": f"Failed to get embedding for text: {e}"}]
        else:
            try:
                q = collection.query(
                    query_embeddings=[embedding],
                    n_results=top_k,
                    where={"$and": [
                        {"user_id": {"$eq": user_id}},
                        {"type": {"$eq": "text"}}
                    ]},
                )
                ids = q.get("ids", [[]])[0]
                distances = q.get("distances", [[]])[0]
                metadatas = q.get("metadatas", [[]])[0]
                out = []
                for i, m in enumerate(ids):
                    meta = metadatas[i] if i < len(metadatas) else {}
                    score = None
                    if distances and i < len(distances) and isinstance(distances[i], (int, float)):
                        score = 1 - distances[i]
                    out.append({
                        "id": m,
                        "score": score if score is not None else 0.0,
                        "metadata": meta,
                        "text": (meta or {}).get("text"),
                    })
                results["text_results"] = out
            except Exception as e:
                results["text_results"] = [{"error": f"Error querying Chroma for text: {e}"}]

    import json
    if not image_bytes and not text:
        return json.dumps({"error": "No text or image provided for retrieval."})

    return json.dumps(results)
