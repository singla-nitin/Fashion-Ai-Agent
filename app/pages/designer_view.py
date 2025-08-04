import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.schema import HumanMessage
from PIL import Image
import io
import base64
from langchain_google_genai import ChatGoogleGenerativeAI  
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("Gemini API key not found. Please set GEMINI_API_KEY in your .env file.")

if "agent_memory" not in st.session_state:
    st.session_state.agent_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GEMINI_API_KEY)
tools = [
    # Add your tool objects here
]

if "langchain_agent" not in st.session_state:
    st.session_state.langchain_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,  # Use Gemini-compatible agent type if available
        memory=st.session_state.agent_memory,
        verbose=True
    )

st.title("🧵 StyloScope Designer Assistant")
st.write("Upload your designs or sketches and chat with your AI assistant.")

uploaded_file = st.file_uploader("Upload a sketch/image", type=["png", "jpg", "jpeg"])
image_bytes = None
base64_image = None

if uploaded_file:
    image_bytes = uploaded_file.read()
    img = Image.open(io.BytesIO(image_bytes))
    st.image(img, width=200, caption="Uploaded Image")
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], list):
            for part in message["content"]:
                if isinstance(part, dict) and part.get("type") == "text":
                    st.write(part["text"])
                elif isinstance(part, dict) and part.get("type") == "image_url":
                    st.image(Image.open(io.BytesIO(base64.b64decode(part["image_url"]["url"].split(',')[1]))), width=200)
        else:
            st.write(message["content"])

task = st.chat_input("Ask your design assistant...")

if task:
    if base64_image:
        user_message_content = [
            {"type": "text", "text": task},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
        st.session_state.messages.append({"role": "user", "content": user_message_content})
        
        st.session_state.agent_memory.chat_memory.add_message(HumanMessage(content=user_message_content))

        agent_input_text = task

    else:
        st.session_state.messages.append({"role": "user", "content": task})
        st.session_state.agent_memory.chat_memory.add_message(HumanMessage(content=task))
        agent_input_text = task

    with st.chat_message("user"):
        if isinstance(st.session_state.messages[-1]["content"], list):
            for part in st.session_state.messages[-1]["content"]:
                if part["type"] == "text":
                    st.write(part["text"])
                elif part["type"] == "image_url":
                    st.image(Image.open(io.BytesIO(base64.b64decode(part["image_url"]["url"].split(',')[1]))), width=200)
        else:
            st.write(st.session_state.messages[-1]["content"])

    with st.spinner("Thinking..."):
        try:
            response = st.session_state.langchain_agent.run(agent_input_text)
        except Exception as e:
            response = f"An error occurred: {e}"
            st.error(response)

    with st.chat_message("assistant"):
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
