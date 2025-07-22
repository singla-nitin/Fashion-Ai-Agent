import streamlit as st
from PIL import Image
import io
import sys
import os
from dotenv import load_dotenv
import base64

# Import LangChain components
from langchain_openai import ChatOpenAI 
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain_core.messages import HumanMessage # For constructing multimodal messages


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
tools_path = os.path.join(project_root, "tools")
sys.path.append(tools_path)


from designer_memory_search import designer_memory_search
from image_memory_search import image_memory_search
from web_search_tool import web_search_tool
from ingest_images import ingest_design_images as add_image_to_memory # Renamed for clarity in tools list
from ingest_text import ingest_design_text as add_text_to_memory # Renamed for clarity in tools list

# Load environment variables (ensure OPENAI_API_KEY is set in your .env file)
dotenv_path = os.path.join(os.getcwd(), '.env')
load_dotenv(dotenv_path)

# Streamlit page config
st.set_page_config(page_title="StyloScope - Designer Chat", layout="wide")

# Sidebar: new session
with st.sidebar:
    st.markdown("## 🎨 StyloScope")
    st.markdown("AI Assistant for Designers")
    if st.button("🆕 New Session"):
        st.session_state.messages = []
        # Clear memory as well for a truly new session
        if "agent_memory" in st.session_state:
            st.session_state.agent_memory.clear() # Clear LangChain memory

# Initialize session storage for chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize LangChain LLM, memory, and agent
# Initialize memory once per session
if "agent_memory" not in st.session_state:
    st.session_state.agent_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

llm = ChatOpenAI(model_name="gpt-4o", temperature=0) # Use gpt-4o
tools = [
    designer_memory_search,
    image_memory_search, 
    web_search_tool,
    add_text_to_memory,
    add_image_to_memory 
]

# Initialize agent once per session
if "langchain_agent" not in st.session_state:
    st.session_state.langchain_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS, # OPENAI_FUNCTIONS is good for tool calling
        memory=st.session_state.agent_memory,
        verbose=True # Set to True for debugging agent's thought process
    )

# Main UI
st.title("🧵 StyloScope Designer Assistant")
st.write("Upload your designs or sketches and chat with your AI assistant.")

# Upload image
uploaded_file = st.file_uploader("Upload a sketch/image", type=["png", "jpg", "jpeg"])
image_bytes = None
base64_image = None

if uploaded_file:
    image_bytes = uploaded_file.read()
    # Display the uploaded image
    img = Image.open(io.BytesIO(image_bytes))
    st.image(img, width=200, caption="Uploaded Image")
    # Encode image to base64 for GPT-4o
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Check if the message content is a list (multimodal)
        if isinstance(message["content"], list):
            for part in message["content"]:
                if isinstance(part, dict) and part.get("type") == "text":
                    st.write(part["text"])
                elif isinstance(part, dict) and part.get("type") == "image_url":
                    # For displaying assistant-generated images or if we store user images
                    # This part might need refinement based on how you store/retrieve images in history
                    st.image(Image.open(io.BytesIO(base64.b64decode(part["image_url"]["url"].split(',')[1]))), width=200) # Decode base64 for display
        else:
            st.write(message["content"])

# Chat input
task = st.chat_input("Ask your design assistant...")

if task:
    # Prepare agent input: create multimodal message if image is uploaded
    if base64_image:
        # Construct the multimodal message for the LLM
        # This is how you pass image data to gpt-4o via LangChain
        user_message_content = [
            {"type": "text", "text": task},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
        # Append to session state messages for display
        st.session_state.messages.append({"role": "user", "content": user_message_content})

        # When you call agent.run(), it expects a string.
        # We need to ensure the agent receives the image.
        # For an agent with tools, the agent's "thought" process itself
        # typically operates on a string prompt. The image is passed to the LLM.
        # A common pattern is to make the agent aware of the image through the prompt
        # and then have a tool to "ingest" it if needed for memory.
        # For direct GPT-4o interpretation, the image goes directly with the message.

        # The agent.run() method directly processes the input string.
        # To make gpt-4o "see" the image *and* use tools, we need a slight adjustment.
        # LangChain's agents typically don't pass multimodal messages *directly* to agent.run().
        # Instead, the LLM within the agent is configured to handle multimodal.

        # Here's the strategy:
        # 1. User provides text and image.
        # 2. We use 'add_image_to_memory' tool to ingest the image into your memory store.
        # 3. We create a multimodal HumanMessage and add it to the memory buffer directly,
        #    so the LLM (gpt-4o) used by the agent can "see" it for its next turn.
        # 4. We then pass the text query to agent.run().

        # First, add the image to memory via your tool (if intended for long-term storage)
        try:
            # Assuming add_image_to_memory expects bytes and might return a confirmation
            with st.spinner("Processing image memory..."):
                ingestion_response = add_image_to_memory.run({"image_bytes": image_bytes, "description": task})
                # You might want to display this ingestion_response or log it
                st.info(f"Image ingestion status: {ingestion_response}")
        except Exception as e:
            st.error(f"Error ingesting image to memory: {e}")

        # Now, prepare the *actual* message to be processed by the LLM within the agent
        # We need to manually add this multimodal message to the conversation history
        # that the agent's LLM will see.
        st.session_state.agent_memory.chat_memory.add_message(HumanMessage(content=user_message_content))

        # Pass the text part of the query to the agent for tool execution
        agent_input_text = task

    else:
        # If no image, just a regular text message
        st.session_state.messages.append({"role": "user", "content": task})
        st.session_state.agent_memory.chat_memory.add_message(HumanMessage(content=task)) # Add to agent memory
        agent_input_text = task

    # Display user message in the chat area (already appended above)
    with st.chat_message("user"):
        # The content has already been appended to st.session_state.messages
        # We just need to ensure it's rendered correctly if it's a list.
        if isinstance(st.session_state.messages[-1]["content"], list):
            for part in st.session_state.messages[-1]["content"]:
                if part["type"] == "text":
                    st.write(part["text"])
                elif part["type"] == "image_url":
                    # This is for displaying the *user's* uploaded image in chat history
                    st.image(Image.open(io.BytesIO(base64.b64decode(part["image_url"]["url"].split(',')[1]))), width=200)
        else:
            st.write(st.session_state.messages[-1]["content"])


    # Run agent
    with st.spinner("Thinking..."):
        try:
            # The agent will now process the text input.
            # Its internal LLM (gpt-4o) will have access to the full chat history
            # including the multimodal message we manually added to agent_memory.chat_history.
            response = st.session_state.langchain_agent.run(agent_input_text)
        except Exception as e:
            response = f"An error occurred: {e}"
            st.error(response)

    # Display assistant mes
    with st.chat_message("assistant"):
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
