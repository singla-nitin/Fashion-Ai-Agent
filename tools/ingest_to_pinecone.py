import os
from dotenv import load_dotenv
import pinecone
from langchain.tools import tool
from PIL import Image
import io
import uuid
import torch
import open_clip



# Load Pinecone API key from .env
load_dotenv()
PINECONE_API_KEY = os.getenv("pinecone_api_key")
PINECONE_ENV = os.getenv("pinecone_env", "us-east-1-aws")  # Default region, change if needed

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

INDEX_NAME = "designer-multimodal-index"
if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(name=INDEX_NAME, dimension=512, metric="cosine")  # 512 for CLIP, 1536 for text-embedding-ada-002
index = pinecone.Index(INDEX_NAME)



device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_e16")
tokenizer = open_clip.get_tokenizer("ViT-B-32")
clip_model = clip_model.to(device)

@tool("ingest_to_pinecone", return_direct=True)
def ingest_to_pinecone(user_id: str, text: str = None, image_bytes: bytes = None, metadata: dict = None) -> str:
    """
   This tool is used to ingest text or image data into a Pinecone index. It requires a user ID and can also accept text, image bytes, and optional metadata. 
   It uses the model's internal embedding capabilities for both text and images. It should be called when you want to store user-related text or image data in Pinecone for searching or retrieval
    """
    if metadata is None:
        metadata = {}
    metadata["user_id"] = user_id
    vector_id = str(uuid.uuid4())

    if image_bytes:
        # Get CLIP embedding for image
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = clip_model.encode_image(image_input).cpu().numpy().tolist()[0]
        except Exception as e:
            return f"Failed to get CLIP embedding for image: {e}"
        metadata["type"] = "image"
        index.upsert([(vector_id, embedding, metadata)])
        return f"Image ingested with id: {vector_id}"
    if text:
        # Use Pinecone's internal embedding model for text
        embed_response = pinecone.embeddings.generate(
            model="text-embedding-ada-002",
            texts=[text]
        )
        embedding = embed_response["data"][0]["embedding"]
        metadata["type"] = "text"
        index.upsert([(vector_id, embedding, metadata)])
        return f"Text ingested with id: {vector_id}"
    else:
        return "No text or image provided."
