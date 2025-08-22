import os
from dotenv import load_dotenv
from langchain.tools import tool
import base64
import requests
from PIL import Image
from io import BytesIO
import streamlit as st
import time

# Load API keys from .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

def enhance_design_prompt(prompt: str, image_context: str = None) -> str:
    """
    Enhance the basic prompt for image generation
    """
    enhanced_prompt = f"high quality professional fashion design, {prompt}, studio lighting, clean background, detailed fabric texture, fashion photography, 8k resolution, professional quality"
    
    if image_context:
        enhanced_prompt += f", inspired by: {image_context}"
    
    return enhanced_prompt

def generate_image_huggingface(prompt: str) -> bytes:
    """Generate image using Hugging Face Stable Diffusion"""
    API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    
    payload = {
        "inputs": prompt,
        "options": {"wait_for_model": True}
    }
    
    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Hugging Face API error: {response.status_code}")

def generate_image_pollinations(prompt: str) -> bytes:
    """Generate image using Pollinations AI (free, no API key needed)"""
    # Clean and encode the prompt for URL
    import urllib.parse
    encoded_prompt = urllib.parse.quote(prompt)
    
    # Pollinations AI endpoint
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&model=flux&enhance=true"
    
    response = requests.get(url, timeout=30)
    
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Pollinations AI error: {response.status_code}")

@tool("generate_design_image", return_direct=True)
def generate_design_image(prompt: str, user_id: str = None) -> str:
    """
    Generate an actual fashion design image using AI image generation services.
    Args:
        prompt: Description of the design to generate.
        user_id: The ID of the user making the request (optional).
    Returns:
        str: A formatted message with the actual generated image
    """
    
    try:
        image_context = None
        
        # Get image context if an image is uploaded
        if hasattr(st.session_state, 'current_image_bytes') and st.session_state.current_image_bytes:
            try:
                import google.generativeai as genai
                genai.configure(api_key=GOOGLE_API_KEY)
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                # Convert bytes to PIL Image for analysis
                image_data = base64.b64decode(st.session_state.current_image_bytes)
                image = Image.open(BytesIO(image_data))
                
                # Analyze the uploaded image for context
                analysis_response = model.generate_content([
                    "Describe this fashion item in 10 words or less for image generation: style, colors, key features",
                    image
                ])
                image_context = analysis_response.text.strip()
            except Exception as e:
                print(f"Could not analyze uploaded image: {e}")
        
        # Enhance the prompt for better image generation
        enhanced_prompt = enhance_design_prompt(prompt, image_context)
        
        image_bytes = None
        
        # Try Pollinations AI first (free, no API key needed)
        try:
            image_bytes = generate_image_pollinations(enhanced_prompt)
        except Exception as e:
            print(f"Pollinations AI failed: {e}")
            
            # Fallback to Hugging Face if available
            if HUGGINGFACE_API_KEY:
                try:
                    image_bytes = generate_image_huggingface(enhanced_prompt)
                except Exception as e:
                    print(f"Hugging Face failed: {e}")
        
        if image_bytes:
            # Convert to base64 for display
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Create response with actual image
            response_message = f"""🎨 **Fashion Design Generated!**

![Generated Design](data:image/png;base64,{image_b64})

**Original Request:** {prompt}
"""
            
            if image_context:
                response_message += f"**Inspired by uploaded image:** {image_context}\n"
            
            response_message += f"\n**Enhanced Prompt Used:** {enhanced_prompt}\n\n✨ Your custom fashion design is ready!"
            
            return response_message
        else:
            return f"❌ Sorry, I couldn't generate an image right now. All image generation services are currently unavailable. Please try again later.\n\n**Your request:** {prompt}"
        
    except Exception as e:
        return f"❌ Error generating image: {str(e)}\n\n**Your request:** {prompt}"