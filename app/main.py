

import streamlit as st
from PIL import Image

# App Configuration
st.set_page_config(page_title="StyloScope - AI Fashion Agent", layout="centered")


# logo = Image.open("assets/logo.png")
# st.image(logo, width=150)

st.markdown("""
    <h1 style='text-align: center; color: #6C3483;'>👗 StyloScope</h1>
    <h3 style='text-align: center; color: #34495E;'>Your Personal AI Assistant in Fashion</h3>
""", unsafe_allow_html=True)

st.markdown("---")

st.markdown("### Who are you?")
user_type = st.selectbox("Choose your role:", ["-- Select --", "Designer", "User"])

if user_type == "Designer":
    st.success("Welcome, Creative Designer! Let's build your fashion vision.")
    if st.button("Enter Designer Workspace"):
        st.switch_page("pages/designer_view.py")  # requires Streamlit >= 1.22

elif user_type == "User":
    st.success("Welcome, Fashion Enthusiast! Let's find your style.")
    if st.button("Enter User Hub"):
        st.switch_page("pages/user_view.py")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: grey;'>Built with ❤️ using LangChain & OpenAI</p>",
    unsafe_allow_html=True
)
