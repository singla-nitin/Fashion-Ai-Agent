import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.schema import HumanMessage, AIMessage
from PIL import Image
import io
import base64
from langchain_google_genai import ChatGoogleGenerativeAI  
from dotenv import load_dotenv
from tools.extract_color_palette import extract_color_palette
from tools.generate_design_image import generate_design_image
from tools.find_fashion_trends import web_search as find_fashion_trends
from tools.ingest_to_pinecone import ingest_to_pinecone


# Load environment variables from .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("Gemini API key not found. Please set GEMINI_API_KEY in your .env file.")
    st.stop()

# Ensure user is authenticated
if "user_id" not in st.session_state or not st.session_state.user_id:
    st.warning("You must be logged in to access the Designer Assistant. Please return to the main page and log in.")
    st.stop()

# Ensure messages is initialized
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": (
                f"You are StyloScope, an expert AI fashion design assistant. Your primary role is to analyze fashion designs, "
                f"provide creative feedback, and help with design improvements. Follow these guidelines:"
                f"\n1. For Design Generation:"
                f"\n   - Always consider the color palette from reference images"
                f"\n   - Create detailed prompts that incorporate color schemes and style elements"
                f"\n   - Maintain consistency with reference images and user preferences"
                f"\n2. For Image Analysis:"
                f"\n   - Use the Color Palette Extractor to analyze designs"
                f"\n   - Provide specific observations about colors, patterns, and style"
                f"\n   - Suggest improvements based on current fashion trends"
                f"\n3. For Design Improvements:"
                f"\n   - Combine user requirements with extracted design elements"
                f"\n   - Use available tools in sequence for better results"
                f"\n   - Explain your design decisions and suggestions"
                f"\n\nYou have access to these specialized fashion tools:"
                f"\n- Color Palette Extractor: Use to analyze and extract color schemes from designs"
                f"\n- Design Image Generator: Create new design variations or concepts"
                f"\n- Fashion Trends Finder: Research current fashion trends"
                f"\n- Pinecone Ingest: Store and index designs for future reference"
                f"\n\nYou are currently assisting user with ID: {st.session_state.user_id}"
                f"\n(Note: The user_id parameter is handled automatically in tool calls)"
            )
        }
    ]

if "agent_memory" not in st.session_state:
    st.session_state.agent_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="input",
        output_key="output"
    )

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    convert_system_message_to_human=True
)

# Import required LangChain components
from langchain.tools import StructuredTool
from typing import Optional, Union, Dict, Any

# Create wrapped versions of each tool
def create_tool_with_user_id(tool):
    from pydantic import BaseModel, Field
    from typing import Optional
    
    # Create a dynamic schema based on the tool's parameters
    class ToolSchema(BaseModel):
        prompt: Optional[str] = Field(None, description="The prompt for generating designs or searching")
        tool_input: Optional[Union[str, dict, bytes]] = Field(None, description="Input for the tool")
        
    def wrapped_sync(prompt: str = None, tool_input: Union[str, dict, bytes] = None, **kwargs):
        # Add user_id to kwargs
        final_input = {'user_id': st.session_state.user_id}
        
        # Handle prompt-based tools (like generate_design_image)
        if hasattr(tool, 'args_schema') and 'prompt' in (tool.args_schema.__annotations__ or {}):
            if prompt:
                final_input['prompt'] = prompt
            elif tool_input and isinstance(tool_input, str):
                final_input['prompt'] = tool_input
        # Handle other tools
        if tool_input is not None:
            if isinstance(tool_input, dict):
                final_input.update(tool_input)
            else:
                final_input['text'] = tool_input
        
        # Add any additional kwargs
        final_input.update(kwargs)
        
        # Ensure we pass the input as tool_input parameter
        return tool.run(tool_input=final_input)

    # Create a new StructuredTool with the wrapped function
    return StructuredTool(
        name=tool.name,
        description=tool.description,
        func=wrapped_sync,
        args_schema=ToolSchema,
        return_direct=True
    )

# Add your tool objects here with automatic user_id injection
tools = [
    create_tool_with_user_id(tool) for tool in [
        find_fashion_trends,
        extract_color_palette,
        generate_design_image,
        ingest_to_pinecone
    ]
]

# Ensure the agent is reinitialized with the current memory state
st.session_state.langchain_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
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

for idx, message in enumerate(st.session_state.messages):
    # Skip system messages in the display
    if message["role"] == "system":
        continue
        
    with st.chat_message(message["role"]):
        if isinstance(message["content"], list):
            # Only show image for the first message that contains it
            is_first_image_message = all(
                not isinstance(prev_msg.get("content", []), list) or
                not any(part.get("type") == "image_url" for part in prev_msg["content"])
                for prev_msg in st.session_state.messages[:idx]
            )
            
            for part in message["content"]:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        st.write(part["text"])
                    elif part.get("type") == "image_url" and is_first_image_message:
                        st.image(Image.open(io.BytesIO(base64.b64decode(part["image_url"]["url"].split(',')[1]))), width=200)
        else:
            st.write(message["content"])

task = st.chat_input("Ask your design assistant...")

if task:
    if base64_image:
        # Create content for Streamlit display
        user_message_content = [
            {"type": "text", "text": task},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
        # Add to Streamlit messages for display
        st.session_state.messages.append({"role": "designer", "content": user_message_content})
        # Add to agent memory as a formatted string
        memory_content = f"{task} [Image attached]"
        st.session_state.agent_memory.chat_memory.add_message(HumanMessage(content=memory_content))
        agent_input_text = task
    else:
        st.session_state.messages.append({"role": "designer", "content": task})
        st.session_state.agent_memory.chat_memory.add_message(HumanMessage(content=task))
        agent_input_text = task

    with st.chat_message("designer"):
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
            # Include the full conversation history and image context
            full_context = "Previous conversation and context:\n"
            
            # Add conversation history
            for msg in st.session_state.messages:
                if msg["role"] != "system":
                    content = msg["content"]
                    if isinstance(content, list):
                        content = " ".join([part["text"] for part in content if part.get("type") == "text"])
                    full_context += f"{msg['role']}: {content}\n"
            
            agent_input = f"{full_context}\nCurrent request: {agent_input_text}"
            
            # Process image-related tasks
            color_palette_info = None
            if base64_image:
                # Always extract color palette for reference
                image_bytes = base64.b64decode(base64_image)
                try:
                    color_palette = extract_color_palette.run(tool_input={"image_bytes": image_bytes})
                    color_palette_info = color_palette
                    color_desc = "Color palette from image: " + ", ".join(color_palette.get("hex_colors", []))
                    agent_input += f"\n{color_desc}"
                except Exception as e:
                    st.error(f"Error analyzing color palette: {e}")
                
                # Add image details and instructions to context
                agent_input += "\nInstructions for design generation:"
                agent_input += "\n1. Use the extracted color palette for design consistency"
                agent_input += "\n2. Consider the style elements from the reference image"
                agent_input += "\n3. When using generate_design_image, incorporate both user requirements and image elements"
            
            # Enhance the prompt if it's a design generation request
            if "generate" in agent_input_text.lower() or "create" in agent_input_text.lower():
                agent_input += "\nNote: When generating designs:"
                agent_input += "\n- First analyze any available color palette"
                agent_input += "\n- Create a detailed prompt that includes color preferences"
                agent_input += "\n- Consider both user requirements and reference image style"
                if color_palette_info:
                    agent_input += f"\n- Use these specific colors: {color_palette_info.get('hex_colors', [])}"
            
            response = st.session_state.langchain_agent.run(agent_input)
            
            # Store important image information in Pinecone if it's a sketch or design
            if base64_image and ("sketch" in agent_input_text.lower() or "design" in agent_input_text.lower()):
                try:
                    ingest_to_pinecone.run(tool_input={
                        "text": f"Image description: {agent_input_text}\nContext: {response}",
                        "image_bytes": base64.b64decode(base64_image)
                    })
                except Exception as e:
                    st.warning(f"Failed to store image context: {str(e)}")
            
            # Format the response
            formatted_response = str(response)
            
            # Check if response contains image data URL and extract it for display
            image_data_url = None
            if "data:image/png;base64," in formatted_response:
                import re
                pattern = r'data:image/png;base64,[A-Za-z0-9+/=]+'
                match = re.search(pattern, formatted_response)
                if match:
                    image_data_url = match.group(0)
            
            # Add response to chat
            with st.chat_message("assistant"):
                if image_data_url:
                    # Display the message text
                    clean_text = formatted_response.replace(f"![Generated Design]({image_data_url})", "")
                    st.write(clean_text)
                    # Display the image
                    try:
                        image_b64 = image_data_url.split(',')[1]
                        st.image(Image.open(io.BytesIO(base64.b64decode(image_b64))), width=400, caption="Generated Design")
                    except Exception as e:
                        st.error(f"Error displaying image: {e}")
                else:
                    st.write(formatted_response)
                    
            # Store in session state (with image data for future reference)
            if image_data_url:
                # Store as structured content for display consistency
                content_with_image = [
                    {"type": "text", "text": formatted_response.replace(f"![Generated Design]({image_data_url})", "").strip()},
                    {"type": "image_url", "image_url": {"url": image_data_url}}
                ]
                st.session_state.messages.append({"role": "assistant", "content": content_with_image})
            else:
                st.session_state.messages.append({"role": "assistant", "content": formatted_response})
            
            # Store in agent memory (as clean text)
            memory_content = formatted_response
            if image_data_url:
                memory_content = memory_content.replace(f"![Generated Design]({image_data_url})", "[Generated image attached]")
                
            st.session_state.agent_memory.chat_memory.add_message(AIMessage(content=memory_content))
            
        except Exception as e:
            error_message = str(e)
            if isinstance(e, dict) and 'error' in e:
                error_message = str(e['error'])
            st.error(f"An error occurred: {error_message}")
            with st.chat_message("assistant"):
                st.write(f"An error occurred: {error_message}")
            formatted_error = f"An error occurred: {error_message}"
            st.session_state.messages.append({"role": "assistant", "content": formatted_error})
            st.session_state.agent_memory.chat_memory.add_message(AIMessage(content=formatted_error))
