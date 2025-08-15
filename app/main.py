import streamlit as st
from PIL import Image
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'db')))
from users import register_user, authenticate_user

# App Configuration
st.set_page_config(page_title="StyloScope - AI Fashion Agent", layout="centered")



st.markdown("""
    <h1 style='text-align: center; color: #6C3483;'>👗 StyloScope</h1>
    <h3 style='text-align: center; color: #34495E;'>Your Personal AI Assistant in Fashion</h3>
""", unsafe_allow_html=True)

st.markdown("---")

st.markdown("### Who are you?")
user_type = st.selectbox("Choose your role:", ["-- Select --", "Designer", "User"])

if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "login"

if user_type in ["Designer", "User"]:
    st.success(f"Welcome, {user_type}! Please log in or register.")
    st.session_state.auth_mode = st.radio("Select mode:", ["login", "register"], horizontal=True, key="auth_mode_radio")
    email = st.text_input("Email", key="email")
    password = st.text_input("Password", type="password", key="password")
    if st.session_state.auth_mode == "register":
        if st.button("Register"):
            user_id = register_user(email, password)
            if user_id:
                st.success("Registration successful! Please log in.")
            else:
                st.error("Email already exists. Please log in or use another email.")
    else:
        if st.button("Login"):
            user_id = authenticate_user(email, password)
            if user_id:
                st.session_state.user_id = user_id
                st.success("Login successful! Redirecting...")
                # Auto-redirect based on user type
                if user_type == "Designer":
                    st.switch_page("pages/designer_view.py")
                elif user_type == "User":
                    st.switch_page("pages/user_view.py")
            else:
                st.error("Invalid credentials. Please try again.")

    # Show workspace info if not yet logged in
    if not st.session_state.user_id and user_type in ["Designer", "User"]:
        st.info(f"After logging in, you'll be automatically redirected to the {user_type} workspace.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: grey;'>Built with ❤️ using LangChain & Gemini</p>",
    unsafe_allow_html=True
)
