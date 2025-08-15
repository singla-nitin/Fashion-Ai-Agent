import io
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from langchain.tools import tool

@tool("extract_color_palette", return_direct=True)
def extract_color_palette(image_bytes: bytes = None, num_colors: int = 5, user_id: str = None) -> dict:
    """
    Extracts the dominant colors from an uploaded image and returns a list of hex color codes.
    Only use this tool when a user has uploaded an image and specifically asks for color extraction or palette analysis.
    
    Args:
        image_bytes: The image file as bytes (optional - will use uploaded image if available).
        num_colors: Number of dominant colors to extract (default 5).
        user_id: The ID of the user making the request (automatically handled).
    Returns:
        dict: {
            "hex_colors": [list of hex color codes],
            "user_id": str,
            "num_colors": int
        }
    """
    if not image_bytes:
        return {"error": "No image provided. Please upload an image first to extract colors."}
    
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((150, 150))  # Resize for speed
        pixels = np.array(image).reshape(-1, 3)
        kmeans = KMeans(n_clusters=num_colors, n_init=10)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_.astype(int)
        hex_colors = ['#%02x%02x%02x' % tuple(color) for color in colors]
        return {"hex_colors": hex_colors, "user_id": user_id, "num_colors": num_colors}
    except Exception as e:
        return {"error": f"Failed to extract color palette: {e}"}
