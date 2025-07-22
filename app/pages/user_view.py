

import streamlit as st
from PIL import Image
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="StyloScope - Style Chat", layout="wide")

st.title("👗 StyloScope - Personal Stylist")
st.write("Upload your outfit or selfie and get personalized fashion advice!")

# Session state for chat
if "customer_messages" not in st.session_state:
    st.session_state.customer_messages = []

# Image input
uploaded_image = st.file_uploader("Upload your outfit or photo", type=["png", "jpg", "jpeg"])

# Text input
user_input = st.chat_input("Ask your stylist anything...")

if user_input or uploaded_image:
    st.session_state.customer_messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)
        if uploaded_image:
            st.image(uploaded_image, width=200)

    messages = [{"role": "system", "content": "You are a fashion stylist helping users pick outfits or improve their look."}]
    for msg in st.session_state.customer_messages:
        messages.append(msg)

    if uploaded_image:
        image_bytes = uploaded_image.read()
        response = openai.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": user_input},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_bytes.decode('latin1')}" }}
                ]}
            ],
            max_tokens=500
        )
        bot_reply = response.choices[0].message.content
    else:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=500
        )
        bot_reply = response.choices[0].message.content

    st.chat_message("assistant").write(bot_reply)
    st.session_state.customer_messages.append({"role": "assistant", "content": bot_reply})
