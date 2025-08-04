import os
from dotenv import load_dotenv
import pinecone
from langchain.tools import tool
from PIL import Image
import io
import torch
import open_clip

# Load Pinecone API key from .env
load_dotenv()
PINECONE_API_KEY = os.getenv("pinecone_api_key")
PINECONE_ENV = os.getenv("pinecone_env", "us-east-1-aws")

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

INDEX_NAME = "designer-multimodal-index"
index = pinecone.Index(INDEX_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_e16")
clip_model = clip_model.to(device)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

@tool("retrieve_from_pinecone", return_direct=True)
def retrieve_from_pinecone(user_id: str, text: str = None, image_bytes: bytes = None, top_k: int = 5) -> str:
    """
    Retrieve the most similar items from Pinecone for a specific user given a text or image query. Returns top_k results with metadata.
    Provide either text or image_bytes. Image queries use CLIP, text queries use text-embedding-ada-002. Only returns results for the given user_id.
    """
    if image_bytes:
        # Get CLIP embedding for image
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = clip_model.encode_image(image_input).cpu().numpy().tolist()[0]
        except Exception as e:
            return f"Failed to get CLIP embedding for image: {e}"
    elif text:
        # Use Pinecone's internal embedding model for text
        embed_response = pinecone.embeddings.generate(
            model="text-embedding-ada-002",
            texts=[text]
        )
        embedding = embed_response["data"][0]["embedding"]
    else:
        return "No text or image provided for retrieval."

    # Query Pinecone
    try:
        query_response = index.query(vector=embedding, top_k=top_k*2, include_metadata=True)  # fetch more to allow filtering
        matches = query_response.get("matches", [])
        user_matches = [
            match for match in matches
            if match.get("metadata", {}).get("user_id") == user_id
        ]
        if not user_matches:
            return f"No similar items found for user_id: {user_id}."
        results = []
        for match in user_matches[:top_k]:
            meta = match.get("metadata", {})
            score = match.get("score", 0)
            results.append(f"ID: {match.get('id')} | Score: {score:.3f} | Metadata: {meta}")
        return "\n".join(results)
    except Exception as e:
        return f"Error querying Pinecone: {e}"