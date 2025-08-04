import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain.tools import tool
from PIL import Image
import io
import uuid
import torch
import open_clip



# Load Pinecone API key from .env
load_dotenv()
PINECONE_API_KEY = os.getenv("pinecone_api_key")
PINECONE_ENV = os.getenv("pinecone_env", "us-east-1")  # Default region, change if needed


# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "designer-multimodal-index"
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1024,  # Match your actual index dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region=PINECONE_ENV  # should be 'us-east-1'
        )
    )
index = pc.Index(INDEX_NAME)



device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_e16")
tokenizer = open_clip.get_tokenizer("ViT-B-32")
clip_model = clip_model.to(device)

@tool("ingest_to_pinecone", return_direct=True)
def ingest_to_pinecone(user_id: str, text: str = None, image_bytes: bytes = None, metadata: dict = None) -> str:
    """
    This tool is used to ingest text or image data into a Pinecone index. It requires a user ID and can also accept text, image bytes, and optional metadata. 
    It uses CLIP model for images and llama-text-embed-v2 for text embeddings. It should be called when you want to store user-related text or image data in Pinecone for searching or retrieval.
    """
    if metadata is None:
        metadata = {}
    metadata["user_id"] = user_id
    vector_id = str(uuid.uuid4())

    try:
        if image_bytes is not None:
            # Process image with CLIP
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = clip_model.encode_image(image_input).cpu().numpy().tolist()[0]
            metadata["type"] = "image"
            
        elif text is not None:
            # Use llama-text-embed-v2 for text
            embed_response = pc.embeddings.generate(
                model="llama-text-embed-v2",
                texts=[text]
            )
            embedding = embed_response["data"][0]["embedding"]
            metadata["type"] = "text"
            metadata["content"] = text
            
        else:
            return "No text or image provided."

        # Upsert to Pinecone
        index.upsert(vectors=[(vector_id, embedding, metadata)])
        return f"{metadata['type'].capitalize()} ingested with id: {vector_id}"

    except Exception as e:
        return f"Failed to process and ingest data: {str(e)}"
