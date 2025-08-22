import os
from dotenv import load_dotenv
from langchain.tools import tool
from PIL import Image
import io
import uuid
import torch
import open_clip
from typing import Optional

# Disable ChromaDB telemetry to prevent errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Local vector store (ChromaDB)
try:
    # Newer Chroma API (>=0.4)
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

# Configure Chroma persistence directory under local db/
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHROMA_DIR = os.path.join(BASE_DIR, "db", "chroma")
os.makedirs(CHROMA_DIR, exist_ok=True)

COLLECTION_NAME = "designer-multimodal-index"

def _get_chroma_collection():
    """Return a Chroma collection, creating it if needed. Raises a clear error if Chroma isn't installed."""
    if _CHROMA_MODE == "persistent":
        try:
            client = PersistentClient(path=CHROMA_DIR)  # type: ignore
            # Use cosine space for CLIP embeddings
            col = client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            return col
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Chroma PersistentClient: {e}")
    else:
        if chromadb is None or Settings is None:
            raise RuntimeError(
                "ChromaDB is not installed. Please install 'chromadb' in your environment."
            )
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
# Lazy init models to avoid slow import on import-time
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


@tool("ingest_to_chroma", return_direct=True)
def ingest_to_chroma(user_id: str, text: str = None, image_bytes: bytes = None, metadata: dict = None) -> str:
    """
    Ingest text or image data into a local ChromaDB collection stored under db/chroma. It requires a user ID, and accepts text, image bytes, and optional metadata.
    Embeddings are computed with CLIP (ViT-B-32, laion2b_e16) for both text and images, normalized for cosine similarity.
    """
    if metadata is None:
        metadata = {}
    metadata["user_id"] = user_id
    vector_id = str(uuid.uuid4())

    try:
        collection = _get_chroma_collection()
        if image_bytes is not None:
            # Process image with CLIP
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            clip_model, preprocess = _get_clip()
            image_input = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = clip_model.encode_image(image_input)
                # Normalize and project to 512 if needed
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                embedding = embedding.cpu().numpy().tolist()[0]
            metadata["type"] = "image"
        elif text is not None:
            # Use CLIP text encoder for same 512-dim space
            clip_model, _ = _get_clip()
            tokenizer = _get_tokenizer()
            tokens = tokenizer([text])
            with torch.no_grad():
                txt = tokens.to(device)
                embedding = clip_model.encode_text(txt)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                embedding = embedding.cpu().numpy().tolist()[0]
            metadata["type"] = "text"
            metadata["text"] = text
        else:
            return "No text or image provided."

        # Add to Chroma
        collection.add(ids=[vector_id], embeddings=[embedding], metadatas=[metadata])
        # Persist is automatic in PersistentClient; legacy client persists on add
        return f"{metadata['type'].capitalize()} ingested locally with id: {vector_id}"

    except Exception as e:
        return f"Failed to process and ingest data locally: {str(e)}"
