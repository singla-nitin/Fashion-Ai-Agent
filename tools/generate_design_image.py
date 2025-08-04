import os
import requests
from dotenv import load_dotenv
from langchain.tools import tool
import base64

# Load Google API key from .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("google_api_key")

IMAGEN_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/imagen:generateImage"

@tool("generate_design_image", return_direct=True)
def generate_design_image(prompt: str) -> dict:
    """
    Generate a design image using Google's Imagen model given a text prompt.
    Returns a dictionary with a base64 image string and a downloadable link.
    """
    if not GOOGLE_API_KEY:
        return {"error": "Google API key not found."}
    headers = {"Content-Type": "application/json"}
    params = {"key": GOOGLE_API_KEY}
    data = {
        "prompt": prompt,
        "imageConfig": {
            "width": 512,
            "height": 512
        }
    }
    try:
        response = requests.post(IMAGEN_API_URL, headers=headers, params=params, json=data)
        response.raise_for_status()
        result = response.json()
        # The API returns base64 image(s) in 'images' field
        image_b64 = result["images"][0]["data"]
        download_link = f"data:image/png;base64,{image_b64}"
        return {
            "message": "Image generated successfully.",
            "image_base64": image_b64,
            "download_link": download_link
        }
    except Exception as e:
        return {"error": f"Failed to generate image: {e}"}