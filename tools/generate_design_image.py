import os
from dotenv import load_dotenv
from langchain.tools import tool
import base64
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

# Load Google API key from .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

@tool("generate_design_image", return_direct=True)
def generate_design_image(prompt: str, user_id: str = None) -> str:
    """
    Generate a design image using Google's Gemini model with image generation capabilities.
    Args:
        prompt: Description of the design to generate.
        user_id: The ID of the user making the request (automatically handled).
    Returns:
        str: A formatted message about the image generation with embedded image data
    """
    if not GOOGLE_API_KEY:
        raise ValueError("Google API key not found.")
    
    try:
        # Configure the client with API key
        os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
        client = genai.Client()
        
        # Generate content with image generation
        response = client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )
        
        # Extract image from response
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                # Get the image data
                image_data = part.inline_data.data
                
                # Convert to base64 string
                image_b64 = base64.b64encode(image_data).decode('utf-8')
                download_link = f"data:image/png;base64,{image_b64}"
                
                # Return a formatted string that includes the image data for display
                return f"🎨 **Image Generated Successfully!**\n\n![Generated Design]({download_link})\n\nI've created a design based on your prompt: '{prompt}'. The image has been generated and is displayed above."
        
        # If no image was found in response
        raise ValueError("No image was generated in the response")
        
    except Exception as e:
        raise ValueError(f"Failed to generate image: {e}")