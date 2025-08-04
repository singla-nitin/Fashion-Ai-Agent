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
def retrieve_from_pinecone(user_id: str, text: str = None, image_bytes: bytes = None, top_k: int = 5) -> dict:
    """
    Retrieve the most similar items from Pinecone for a specific user given a text and/or image query.
    Provide either or both text and image_bytes. Image queries use CLIP, text queries use text-embedding-ada-002. Only returns results for the given user_id.

    Returns:
        A dictionary with the following format:
        {
            "image_results": [
                {
                    "id": str,           # Vector ID
                    "score": float,     # Similarity score
                    "metadata": dict,   # Metadata stored with the vector
                    "image_data": str   # Original image data (e.g., base64 or URL) if available
                },
                ...
            ],
            "text_results": [
                {
                    "id": str,           # Vector ID
                    "score": float,     # Similarity score
                    "metadata": dict,   # Metadata stored with the vector
                    "text": str         # Original text if available
                },
                ...
            ]
        }
        If there are errors, the lists will contain error messages instead of result dicts.
    """
    results = {"image_results": [], "text_results": []}

    # Image query
    if image_bytes:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = clip_model.encode_image(image_input).cpu().numpy().tolist()[0]
        except Exception as e:
            results["image_results"] = [{"error": f"Failed to get CLIP embedding for image: {e}"}]
        else:
            try:
                query_response = index.query(
                    vector=embedding,
                    top_k=top_k,
                    include_metadata=True,
                    filter={"user_id": user_id, "type": "image"}
                )
                matches = query_response.get("matches", [])
                results["image_results"] = [
                    {
                        "id": match.get("id"),
                        "score": match.get("score", 0),
                        "metadata": match.get("metadata", {}),
                        "image_data": match.get("metadata", {}).get("image_data")
                    }
                    for match in matches
                ]
            except Exception as e:
                results["image_results"] = [{"error": f"Error querying Pinecone for image: {e}"}]

    # Text query
    if text:
        try:
            embed_response = pinecone.embeddings.generate(
                model="text-embedding-ada-002",
                texts=[text]
            )
            embedding = embed_response["data"][0]["embedding"]
        except Exception as e:
            results["text_results"] = [{"error": f"Failed to get embedding for text: {e}"}]
        else:
            try:
                query_response = index.query(
                    vector=embedding,
                    top_k=top_k,
                    include_metadata=True,
                    filter={"user_id": user_id, "type": "text"}
                )
                matches = query_response.get("matches", [])
                results["text_results"] = [
                    {
                        "id": match.get("id"),
                        "score": match.get("score", 0),
                        "metadata": match.get("metadata", {}),
                        "text": match.get("metadata", {}).get("text")
                    }
                    for match in matches
                ]
            except Exception as e:
                results["text_results"] = [{"error": f"Error querying Pinecone for text: {e}"}]

    if not image_bytes and not text:
        return {"error": "No text or image provided for retrieval."}

    return results