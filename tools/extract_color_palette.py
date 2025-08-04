import io
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from langchain.tools import tool

@tool("extract_color_palette", return_direct=True)
def extract_color_palette(image_bytes: bytes, num_colors: int = 5) -> dict:
    """
    Extracts the dominant colors from an image and returns a list of hex color codes.
    Args:
        image_bytes: The image file as bytes.
        num_colors: Number of dominant colors to extract (default 5).
    Returns:
        dict: {"hex_colors": [list of hex color codes]}
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((150, 150))  # Resize for speed
        pixels = np.array(image).reshape(-1, 3)
        kmeans = KMeans(n_clusters=num_colors, n_init=10)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_.astype(int)
        hex_colors = ['#%02x%02x%02x' % tuple(color) for color in colors]
        return {"hex_colors": hex_colors}
    except Exception as e:
        return {"error": f"Failed to extract color palette: {e}"}
