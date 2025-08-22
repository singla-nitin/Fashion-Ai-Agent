
import sys, os, io, base64
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.schema import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import StructuredTool
from typing import Optional, Union, Dict, Any
# MongoDB for chat history
from pymongo import MongoClient, ASCENDING, DESCENDING
import uuid
from datetime import datetime

# Small helper types
Message = Dict[str, Any]

# Tool imports will be loaded after env vars are configured to avoid telemetry noise



# Load environment variables from .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
# Silence Chroma telemetry noise (incompatible posthog) – best-effort
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "false")
if not GEMINI_API_KEY:
    st.error("Gemini API key not found. Please set GEMINI_API_KEY in your .env file.")
    st.stop()

# Ensure user is authenticated
if "user_id" not in st.session_state or not st.session_state.user_id:
    st.warning("You must be logged in to access the Designer Assistant. Please return to the main page and log in.")
    st.stop()


# MongoDB setup with basic error handling and index hints
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    client.admin.command('ping')  # quick connectivity check
    db = client["styloscope"]
    chats_collection = db["chats"]
    # Ensure helpful indexes (idempotent)
    try:
        chats_collection.create_index([("user_id", ASCENDING), ("timestamp", DESCENDING)])
        chats_collection.create_index([("chat_id", ASCENDING)], unique=True)
    except Exception:
        pass
except Exception as e:
    st.sidebar.error(f"MongoDB connection error: {e}")
    st.stop()

############################
# Chat persistence utilities
############################

# Now safe to import tools (telemetry env vars are set)
from tools.extract_color_palette import extract_color_palette
from tools.generate_design_image import generate_design_image
from tools.find_fashion_trends import web_search as find_fashion_trends
from tools.ingest_to_chroma import ingest_to_chroma
from tools.retrieve_from_chroma import retrieve_from_chroma

def _compute_chat_title(messages: list[Message]) -> str:
    # Use the last designer/user text as title; fallback to first non-system text; else Untitled
    for msg in reversed(messages):
        if msg.get("role") in {"designer", "user"}:
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                        return (part["text"].strip() or "Untitled")[:80]
            elif isinstance(content, str) and content.strip():
                return content.strip()[:80]
    # fallback search any text
    for msg in messages:
        if isinstance(msg.get("content"), str) and msg["content"].strip():
            return msg["content"].strip()[:80]
    return "Untitled"


# Sidebar chat management
def get_user_chats(user_id):
    try:
        return list(
            chats_collection.find({"user_id": user_id}, {"_id": 0, "chat_id": 1, "title": 1, "timestamp": 1})
            .sort("timestamp", DESCENDING)
        )
    except Exception as e:
        st.sidebar.error(f"Failed to load chats: {e}")
        return []

def load_chat(chat_id):
    try:
        chat = chats_collection.find_one({"chat_id": chat_id})
        return chat["messages"] if chat else []
    except Exception as e:
        st.sidebar.error(f"Failed to load chat: {e}")
        return []

def save_chat(user_id, chat_id, messages, title=None):
    try:
        chats_collection.update_one(
            {"chat_id": chat_id},
            {"$set": {
                "user_id": user_id,
                "messages": messages,
                "title": (title or _compute_chat_title(messages)),
                "timestamp": datetime.utcnow(),
            }},
            upsert=True
        )
    except Exception as e:
        st.sidebar.error(f"Failed to save chat: {e}")

######################
# Sidebar chat UI
######################
st.sidebar.title("Chats")

def _reset_memory_with_messages(messages: list[Message]):
    """Reset agent memory and populate it with chat history from MongoDB"""
    st.session_state.agent_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="input",
        output_key="output"
    )
    
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        
        # Skip system messages
        if role == "system":
            continue
            
        # Extract text content from various formats
        text_content = ""
        if isinstance(content, list):
            # Handle multimodal content (text + images)
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                    text_parts.append(part["text"].strip())
            text_content = " ".join(text_parts)
        elif isinstance(content, str):
            text_content = content.strip()
        
        # Add to memory if we have text content
        if text_content:
            if role in {"designer", "user"}:
                st.session_state.agent_memory.chat_memory.add_message(
                    HumanMessage(content=text_content)
                )
            elif role == "assistant":
                st.session_state.agent_memory.chat_memory.add_message(
                    AIMessage(content=text_content)
                )
    
    print(f"Memory reset with {len(st.session_state.agent_memory.chat_memory.messages)} messages")

def add_message_to_memory(role: str, content: Union[str, list]):
    """Add a message to agent memory in real-time"""
    text_content = ""
    
    if isinstance(content, list):
        # Extract text from multimodal content
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                text_parts.append(part["text"].strip())
        text_content = " ".join(text_parts)
    elif isinstance(content, str):
        text_content = content.strip()
    
    # Add to memory if we have text content
    if text_content:
        if role in {"designer", "user"}:
            st.session_state.agent_memory.chat_memory.add_message(
                HumanMessage(content=text_content)
            )
        elif role == "assistant":
            # If content is a dict, convert to string
            if isinstance(content, dict):
                import json
                text_content = json.dumps(content)
            st.session_state.agent_memory.chat_memory.add_message(
                AIMessage(content=text_content)
            )

    print(f"Memory reset with {len(st.session_state.agent_memory.chat_memory.messages)} messages")

def _compute_retrieval_k(query: str) -> int:
    """Simple adaptive top_k: return 3 for most queries, 5 for broad searches"""
    q = (query or "").lower()
    if any(tok in q for tok in ["all", "everything", "complete", "full"]):
        return 5
    return 3

def get_user_info_from_memory(memory, keywords):
    """Check if any recent conversation contains relevant user information"""
    for msg in memory.chat_memory.messages:
        content = msg.content if hasattr(msg, "content") else ""
        if any(kw.lower() in content.lower() for kw in keywords):
            return content
    return None

def retrieve_user_context():
    """Retrieve user preferences and style information from memory first, then ChromaDB."""
    if st.session_state.context_retrieved:
        return ""
    keywords = ['name', 'preferences', 'style', 'favorite', 'design', 'fashion']
    info = get_user_info_from_memory(st.session_state.agent_memory, keywords)
    if info:
        st.session_state.context_retrieved = True
        return f"User context from memory: {info}"
    # No manual ChromaDB call here - let the agent decide
    st.session_state.context_retrieved = True
    return ""


if "chat_id" not in st.session_state:
    st.session_state.chat_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_memory" not in st.session_state:
    st.session_state.agent_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="input",
        output_key="output"
    )
if "context_retrieved" not in st.session_state:
    st.session_state.context_retrieved = False

# System prompt and welcome message (define early so sidebar can use them)
system_message = (
    f"You are StyloScope, an expert AI fashion design assistant with multimodal capabilities. "
    f"You help users create beautiful fashion designs, find trends, analyze colors, and build their personal style collections. "
    f"CAPABILITIES: "
    f"- You CAN analyze uploaded images for themes, styles, colors, and fashion elements "
    f"- You CAN extract color palettes from images using the extract_color_palette tool "
    f"- You CAN generate new fashion designs using generate_design_image tool "
    f"- You CAN search for fashion trends using find_fashion_trends tool "
    f"- You CAN view and analyze any images uploaded by users directly "
    f"CRITICAL MEMORY AND RETRIEVAL BEHAVIOR: "
    f"- You have access to our full conversation history through your chat_history memory - this contains all previous messages between you and the user in this session. "
    f"- ALWAYS reference and use information from our conversation history when relevant. "
    f"- If a user asks for information (like their name, preferences, etc.) that is NOT in the current conversation memory, you MUST automatically use the retrieve_from_chroma tool to check previous sessions. "
    f"- DO NOT ask the user for information that might be stored in ChromaDB - always check there first using retrieve_from_chroma. "
    f"- When users share new personal info, store it with ingest_to_chroma for future sessions. "
    f"AUTOMATIC RETRIEVAL RULES: "
    f"- User asks 'tell me name' or 'what is my name' → Check conversation memory first → If not found, automatically use retrieve_from_chroma with query 'user name preferences' "
    f"- User asks about preferences/style → Check conversation memory first → If not found, automatically use retrieve_from_chroma with relevant query "
    f"- User asks 'use chroma retrieval' → Immediately use retrieve_from_chroma tool without asking for more context "
    f"- NEVER ask users to provide information that could be retrieved from previous sessions "
    f"CONVERSATION CONTEXT: "
    f"- You can see our entire conversation in your memory - use it! "
    f"- If a user shared their name earlier in this chat, you should remember it without asking again. "
    f"- Reference previous topics, preferences, and designs we've discussed. "
    f"TOOL USAGE PRIORITY: "
    f"1. Check conversation memory (chat_history) first for current session info "
    f"2. If information not found in memory, automatically use retrieve_from_chroma to check previous sessions "
    f"3. Only ask the user directly if information is not found in either place "
    f"4. Use ingest_to_chroma to store new user information for future sessions "
    f"You are currently assisting user with ID: {st.session_state.user_id}."
)
welcome_message = "👋 Hello! I'm StyloScope, your AI fashion design assistant. I'm here to help you create stunning designs, explore fashion trends, and build your personal style collection.\n\nWhat would you like to work on today? You can:\n• Upload a sketch or image for me to analyze\n• Ask me to generate design ideas\n• Explore current fashion trends\n• Get color palette suggestions\n• Or simply chat about your style preferences!\n\nHow can I help bring your fashion vision to life? ✨"

user_chats = get_user_chats(st.session_state.user_id)

new_chat_col1, new_chat_col2 = st.sidebar.columns([1, 3])
with new_chat_col1:
    if st.button("＋", key="new_chat_icon", help="New Chat"):
        st.session_state.chat_id = str(uuid.uuid4())
        st.session_state.messages = [
            {"role": "system", "content": system_message},
            {"role": "assistant", "content": welcome_message}
        ]
        st.session_state.context_retrieved = False
        _reset_memory_with_messages(st.session_state.messages)
        save_chat(st.session_state.user_id, st.session_state.chat_id, st.session_state.messages)
        st.rerun()
with new_chat_col2:
    st.markdown("<div style='font-weight:600;margin-top:6px;'>New Chat</div>", unsafe_allow_html=True)

st.sidebar.markdown("""
<style>
.chat-list { max-height: 70vh; overflow-y: auto; }
.chat-item { padding: 8px 10px; margin: 6px 0; border-radius: 6px; background: #f4f4f8; cursor: pointer; }
.chat-item.active { background: #e6f0ff; border-left: 3px solid #3b82f6; }
.chat-title { font-size: 13px; color: #111827; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.chat-time { font-size: 11px; color: #6b7280; }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("<div class='chat-list'>", unsafe_allow_html=True)
for chat in user_chats:
    chat_id = chat.get("chat_id")
    title = chat.get("title") or "Untitled"
    ts = chat.get("timestamp")
    ts_str = "" if not ts else str(ts)[:19].replace("T", " ")
    is_active = (chat_id == st.session_state.chat_id)
    # Render as a button-like block
    block_key = f"chat_block_{chat_id}"
    css_active = 'active' if is_active else ''
    html_block = (
        f"<div class='chat-item {css_active}'>"
        f"<div class='chat-title'>{title}</div>"
        f"<div class='chat-time'>{ts_str}</div>"
        f"</div>"
    )
    st.sidebar.markdown(html_block, unsafe_allow_html=True)
    if st.sidebar.button("Open", key=block_key):
        st.session_state.chat_id = chat_id
        loaded_messages = load_chat(chat_id)
        st.session_state.messages = loaded_messages
        st.session_state.context_retrieved = False
        
        # Reset agent memory with loaded chat history
        _reset_memory_with_messages(loaded_messages)
        
        # Debug info
        print(f"Loaded chat {chat_id} with {len(loaded_messages)} messages")
        print(f"Agent memory now has {len(st.session_state.agent_memory.chat_memory.messages)} messages")
        
        st.rerun()
st.sidebar.markdown("</div>", unsafe_allow_html=True)

# Ensure messages is initialized
# Ensure messages/memory set above


# LLM setup
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    # Remove deprecated parameter - Gemini handles system messages natively
)

# Register all tools for agent (including retrieval)
def create_tool_with_user_id(tool):
    from pydantic import BaseModel, Field
    from typing import Optional
    class ToolSchema(BaseModel):
        prompt: Optional[str] = Field(None, description="Prompt for tool")
        tool_input: Optional[Union[str, dict, bytes]] = Field(None, description="Input for the tool")
    def wrapped_sync(prompt: str = None, tool_input: Union[str, dict, bytes] = None, **kwargs):
        # Map inputs correctly based on tool name
        if tool.name == "web_search":
            # web_search expects: prompt, user_id
            final_input = {
                'prompt': prompt or tool_input or kwargs.get('text', ''),
                'user_id': st.session_state.user_id
            }
        elif tool.name == "generate_design_image":
            # generate_design_image expects: prompt, user_id
            final_input = {
                'prompt': prompt or tool_input or kwargs.get('text', ''),
                'user_id': st.session_state.user_id
            }
        elif tool.name == "extract_color_palette":
            # extract_color_palette expects: image_bytes, user_id
            final_input = {
                'user_id': st.session_state.user_id
            }
            # Check if image_bytes provided directly
            if isinstance(tool_input, bytes):
                final_input['image_bytes'] = tool_input
            elif isinstance(tool_input, dict) and 'image_bytes' in tool_input:
                # Handle case where tool_input is a dict with image_bytes
                if isinstance(tool_input['image_bytes'], bytes):
                    final_input['image_bytes'] = tool_input['image_bytes']
                elif tool_input['image_bytes'] == 'image_from_chat':
                    # Agent is asking for the current uploaded image
                    if 'current_image_bytes' in st.session_state and st.session_state.current_image_bytes:
                        final_input['image_bytes'] = st.session_state.current_image_bytes
                    else:
                        # No image available - return error immediately to avoid loops
                        return {"error": "No image uploaded. Please upload an image first to extract colors."}
                else:
                    return {"error": "Invalid image data provided."}
            else:
                # Use currently uploaded image if available
                if 'current_image_bytes' in st.session_state and st.session_state.current_image_bytes:
                    final_input['image_bytes'] = st.session_state.current_image_bytes
                else:
                    return {"error": "No image uploaded. Please upload an image first to extract colors."}
                # If prompt provided, use it as text context
                if prompt:
                    final_input['text'] = prompt
        elif tool.name in ["ingest_to_chroma", "retrieve_from_chroma"]:
            # ChromaDB tools expect: user_id, text, image_bytes, metadata
            final_input = {'user_id': st.session_state.user_id}
            if isinstance(tool_input, dict):
                final_input.update(tool_input)
            elif isinstance(tool_input, bytes):
                final_input['image_bytes'] = tool_input
            elif isinstance(tool_input, str) or prompt:
                final_input['text'] = prompt or tool_input
            # Simple adaptive top_k for retrieval
            if tool.name == "retrieve_from_chroma":
                txt = final_input.get('text') or ''
                final_input.setdefault('top_k', _compute_retrieval_k(txt))
        else:
            # Default mapping for unknown tools
            final_input = {'user_id': st.session_state.user_id}
            if prompt:
                final_input['prompt'] = prompt
            if tool_input is not None:
                if isinstance(tool_input, dict):
                    final_input.update(tool_input)
                elif isinstance(tool_input, bytes):
                    final_input['image_bytes'] = tool_input
                else:
                    final_input['text'] = tool_input
        
        final_input.update(kwargs)
        return tool.run(tool_input=final_input)
    return StructuredTool(
        name=tool.name,
        description=tool.description,
        func=wrapped_sync,
        args_schema=ToolSchema,
        # Return direct for image generation and web search to preserve formatting/links
        return_direct=(tool.name in ["generate_design_image", "web_search"])
    )

tools = [
    create_tool_with_user_id(tool) for tool in [
        find_fashion_trends,
        extract_color_palette,
        generate_design_image,
        ingest_to_chroma,
        retrieve_from_chroma
    ]
]


# System prompt: let LLM plan tool usage and chaining (already defined above)

if not st.session_state.messages:
    st.session_state.messages = [
        {"role": "system", "content": system_message + "\n\nEXAMPLES:\n- User: 'My name is John' → Response: 'Nice to meet you John!' + ingest_to_chroma\n- User: 'What's my name?' → Check conversation memory first. If 'John' was mentioned earlier in this chat, respond 'Your name is John'. If NOT found in current conversation, automatically use retrieve_from_chroma with query 'user name' to check previous sessions.\n- User: 'tell me name' → Check memory → If not found, automatically use retrieve_from_chroma with query 'user name preferences' → If still not found, then ask user.\n- User: 'use chroma retrieval' → Immediately use retrieve_from_chroma tool with query 'user preferences style fashion' without asking for more context.\n- User: 'What are my preferences?' → Check memory for recent preferences → If not found, automatically use retrieve_from_chroma with query 'user preferences style fashion'."},
        {"role": "assistant", "content": welcome_message}
    ]
    # Initialize agent memory with the welcome message
    _reset_memory_with_messages(st.session_state.messages)
    # Save initial chat
    save_chat(st.session_state.user_id, st.session_state.chat_id, st.session_state.messages)

st.session_state.langchain_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    memory=st.session_state.agent_memory,
    verbose=True,
    agent_kwargs={
        "system_message": system_message,
        "prefix": system_message,  # fallback for templates using prefix
    },
    max_iterations=2,  # Prevent infinite loops - reduced from 3
    early_stopping_method="force",  # Force stop if max iterations reached
    handle_parsing_errors=True,  # Handle parsing errors gracefully
)

# Debug: Show current memory content
if st.sidebar.button("Debug: Show Memory"):
    st.sidebar.write("Current Agent Memory:")
    for msg in st.session_state.agent_memory.chat_memory.messages:
        st.sidebar.write(f"{type(msg).__name__}: {msg.content[:100]}...")

# Debug: Show current messages
if st.sidebar.button("Debug: Show Messages"):
    st.sidebar.write("Current Session Messages:")
    for i, msg in enumerate(st.session_state.messages):
        st.sidebar.write(f"{i}: {msg['role']}: {str(msg['content'])[:100]}...")

# Debug: Test ingest tool manually
if st.sidebar.button("Debug: Test Ingest"):
    try:
        from tools.ingest_to_chroma import ingest_to_chroma
        result = ingest_to_chroma.run(tool_input={
            'user_id': st.session_state.user_id,
            'text': 'Test user information: My name is Nitin Singla',
            'metadata': {'type': 'user_info', 'category': 'name'}
        })
        st.sidebar.success(f"Ingest test result: {result}")
    except Exception as e:
        st.sidebar.error(f"Ingest test failed: {e}")

# Debug: Test retrieve tool manually  
if st.sidebar.button("Debug: Test Retrieve"):
    try:
        from tools.retrieve_from_chroma import retrieve_from_chroma
        result = retrieve_from_chroma.run(tool_input={
            'user_id': st.session_state.user_id,
            'text': 'name Nitin',
            'top_k': 3
        })
        st.sidebar.success(f"Retrieve test result: {result}")
    except Exception as e:
        st.sidebar.error(f"Retrieve test failed: {e}")


st.title("🧵 StyloScope Designer Assistant")
st.write("Upload your designs or sketches and chat with your AI assistant.")

uploaded_file = st.file_uploader("Upload a sketch/image", type=["png", "jpg", "jpeg"])
image_bytes = None
base64_image = None
if uploaded_file:
    image_bytes = uploaded_file.read()
    # Store in session state for tools to access
    st.session_state.current_image_bytes = image_bytes
    img = Image.open(io.BytesIO(image_bytes))
    st.image(img, width=200, caption="Uploaded Image")
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
else:
    # Clear session state if no image uploaded
    st.session_state.current_image_bytes = None

# Display chat history
for idx, message in enumerate(st.session_state.messages):
    if message["role"] == "system":
        continue
    with st.chat_message(message["role"]):
        if isinstance(message["content"], list):
            for part in message["content"]:
                if isinstance(part, dict):
                    if part.get("type") == "text" and part.get("text"):
                        st.write(part["text"])
                    elif part.get("type") == "image_url" and part.get("image_url", {}).get("url"):
                        try:
                            image_url = part["image_url"]["url"]
                            if "base64," in image_url:
                                image_data = base64.b64decode(image_url.split(',')[1])
                                st.image(Image.open(io.BytesIO(image_data)), width=200)
                            else:
                                st.write(f"Image URL: {image_url}")
                        except Exception as e:
                            st.error(f"Could not display image: {e}")
        else:
            st.write(message["content"])

task = st.chat_input("Ask your design assistant...")


if task:
    # Display user message immediately
    with st.chat_message("designer"):
        if base64_image:
            user_message_content = [
                {"type": "text", "text": task},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
            st.write(task)
            try:
                image_data = base64.b64decode(base64_image)
                st.image(Image.open(io.BytesIO(image_data)), width=200)
            except Exception:
                pass
        else:
            st.write(task)
    
    # Add user message to chat history
    image_caption = None
    if base64_image:
        # Note: Image caption generation could be added here in the future
        image_caption = None
        user_message_content = [
            {"type": "text", "text": task},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
        st.session_state.messages.append({"role": "designer", "content": user_message_content})
        
        # Add to agent memory
        add_message_to_memory("designer", user_message_content)
    else:
        st.session_state.messages.append({"role": "designer", "content": task})
        
        # Add to agent memory
        add_message_to_memory("designer", task)
    
    # Persist user turn immediately with updated title
    save_chat(st.session_state.user_id, st.session_state.chat_id, st.session_state.messages)

    # Let the LLM decide when to use tools naturally
    with st.spinner("Thinking..."):
        try:
            # Build agent input with image if available
            if base64_image:
                # For multimodal input, we need to pass both text and image
                agent_input = [
                    {"type": "text", "text": task},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
                if image_caption:
                    agent_input[0]["text"] += f"\nImage description: {image_caption}"
            else:
                # Text only input
                agent_input = task
                if image_caption:
                    agent_input += f"\nImage description: {image_caption}"

            # Add explicit conversation context to help agent understand what's in memory
            memory_messages = st.session_state.agent_memory.chat_memory.messages
            if len(memory_messages) > 1:  # More than just system message
                recent_context = []
                for msg in memory_messages[-6:]:  # Last 6 messages for context
                    role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                    content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                    recent_context.append(f"{role}: {content}")
                
                if recent_context:
                    context_text = f"\n\nRecent conversation context:\n" + "\n".join(recent_context)
                    if isinstance(agent_input, list):
                        # Add context to the text part of multimodal input
                        agent_input[0]["text"] += context_text
                    else:
                        # Add context to text input
                        agent_input += context_text

            # Simply run the agent - it will use memory and tools as needed
            try:
                response = st.session_state.langchain_agent.run(agent_input)
            except Exception as agent_error:
                # Handle agent errors gracefully
                error_msg = str(agent_error)
                if "maximum iterations" in error_msg.lower() or "parsing" in error_msg.lower():
                    response = "I apologize, but I encountered some difficulty processing your request. Could you please try rephrasing your question or upload the image again?"
                else:
                    response = f"I encountered an error: {error_msg}"

            # Display and store response
            formatted_response = response
            if isinstance(formatted_response, dict):
                formatted_response = json.dumps(formatted_response)
            else:
                formatted_response = str(formatted_response)

            # If a tool returned raw JSON, convert to friendly text
            try:
                resp_strip = formatted_response.strip()
                if resp_strip.startswith('{') or resp_strip.startswith('['):
                    import json as _json
                    parsed = _json.loads(resp_strip)
                    if isinstance(parsed, dict) and ("text_results" in parsed or "image_results" in parsed):
                        k = _compute_retrieval_k(agent_input)
                        texts = []
                        for item in (parsed.get("text_results") or [])[:k]:
                            if isinstance(item, dict):
                                t = item.get("text") or item.get("metadata", {}).get("text")
                                if t:
                                    texts.append(t.strip())
                        if texts:
                            formatted_response = "Here’s what I found: \n- " + "\n- ".join(texts)
            except Exception:
                pass
            image_data_url = None
            if "data:image/png;base64," in formatted_response:
                import re
                pattern = r'data:image/png;base64,[A-Za-z0-9+/=]+'
                match = re.search(pattern, formatted_response)
                if match:
                    image_data_url = match.group(0)
            with st.chat_message("assistant"):
                if image_data_url:
                    clean_text = formatted_response.replace(f"![Generated Design]({image_data_url})", "")
                    st.write(clean_text)
                    try:
                        image_b64 = image_data_url.split(',')[1]
                        st.image(Image.open(io.BytesIO(base64.b64decode(image_b64))), width=400, caption="Generated Design")
                    except Exception as e:
                        st.error(f"Error displaying image: {e}")
                else:
                    st.write(formatted_response)
            # Store in session state
            if image_data_url:
                content_with_image = [
                    {"type": "text", "text": formatted_response.replace(f"![Generated Design]({image_data_url})", "").strip()},
                    {"type": "image_url", "image_url": {"url": image_data_url}}
                ]
                st.session_state.messages.append({"role": "assistant", "content": content_with_image})
                # Add to agent memory
                add_message_to_memory("assistant", content_with_image)
            else:
                if isinstance(formatted_response, dict):
                    formatted_response = json.dumps(formatted_response)
                st.session_state.messages.append({"role": "assistant", "content": formatted_response})
                add_message_to_memory("assistant", formatted_response)
            save_chat(st.session_state.user_id, st.session_state.chat_id, st.session_state.messages)
        except Exception as e:
            error_message = str(e)
            st.error(f"An error occurred: {error_message}")
            with st.chat_message("assistant"):
                st.write(f"An error occurred: {error_message}")
            formatted_error = f"An error occurred: {error_message}"
            st.session_state.messages.append({"role": "assistant", "content": formatted_error})
            
            # Add error to agent memory
            add_message_to_memory("assistant", formatted_error)
            
            save_chat(st.session_state.user_id, st.session_state.chat_id, st.session_state.messages)
