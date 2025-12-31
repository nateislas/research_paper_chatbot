"""
Streamlit Frontend for LlamaIndex Document Chatbot
A beautiful web interface for uploading documents and querying them.
"""

import streamlit as st
import os
import sys
import queue
import asyncio
from pathlib import Path
import tempfile
from dotenv import load_dotenv

# Add parent directory to Python path so we can import the package
# __file__ is at: research_paper_chatbot/research_paper_chatbot/app.py
# We need: research_paper_chatbot/ (two levels up)
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load environment variables from project root
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

# Streamlit Cloud uses st.secrets - make it available as environment variable
try:
    if hasattr(st, 'secrets'):
        if 'GOOGLE_API_KEY' in st.secrets:
            os.environ['GOOGLE_API_KEY'] = st.secrets['GOOGLE_API_KEY']
        if 'LLAMA_API_KEY' in st.secrets:
            os.environ['LLAMA_API_KEY'] = st.secrets['LLAMA_API_KEY']
except:
    pass  # Not in Streamlit context or secrets not available

from research_paper_chatbot.workflow import SyncDocumentChatbot, _runner, DocumentChatbotWorkflow

# Page configuration
st.set_page_config(
    page_title="Research Assistant",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

## Custom CSS for Premium Design
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Root Variables - Modern SaaS Palette */
    :root {
        --primary: #3b82f6;       /* Radiant Blue */
        --primary-dark: #2563eb;
        --secondary: #64748b;     /* Slate 500 */
        --accent: #8b5cf6;        /* Violet */
        
        --bg-main: #ffffff;
        --bg-secondary: #f8fafc;  /* Slate 50 */
        --bg-tertiary: #f1f5f9;   /* Slate 100 */
        
        --text-main: #0f172a;     /* Slate 900 */
        --text-sub: #475569;      /* Slate 600 */
        --text-muted: #94a3b8;    /* Slate 400 */
        
        --border-light: #e2e8f0;  /* Slate 200 */
        --border-medium: #cbd5e1; /* Slate 300 */
        
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        --shadow-soft: 0 20px 40px -10px rgba(0,0,0,0.05);
    }

    /* Reset & Base */
    .stApp {
        background-color: var(--bg-secondary);
        font-family: 'Inter', sans-serif;
        color: var(--text-main);
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        letter-spacing: -0.025em;
        color: var(--text-main);
    }
    
    /* Header Cleanup */
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 3rem !important;
    }
    
    /* Main Container Spacing */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 8rem !important; /* Space for fixed chat input */
        max-width: 95% !important; /* Fluid width to use available space */
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-main) !important;
        border-right: 1px solid var(--border-light);
        box-shadow: var(--shadow-sm);
    }
    
    section[data-testid="stSidebar"] .block-container {
        padding: 2rem 1.5rem !important;
    }
    
    /* Custom Sidebar Components */
    .sidebar-widget {
        background: var(--bg-secondary);
        border: 1px solid var(--border-light);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .sidebar-header {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-muted);
        font-weight: 600;
        margin-bottom: 0.75rem;
    }

    .status-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.875rem;
        font-weight: 500;
        color: var(--text-sub);
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
    }
    
    .status-dot.active { background-color: #10b981; box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2); }
    .status-dot.inactive { background-color: #ef4444; box-shadow: 0 0 0 2px rgba(239, 68, 68, 0.2); }

    /* Chat Messages */
    .stChatMessage {
        background-color: transparent !important;
        padding: 0 !important;
        margin-bottom: 1.5rem;
    }
    
    .stChatMessage .stMarkdown {
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Avatars */
    .stChatMessage .stAvatar {
        background-color: transparent !important;
        border: 1px solid var(--border-light);
        width: 2rem !important;
        height: 2rem !important;
    }

    /* Message Bubbles */
    .message-bubble {
        padding: 1rem 1.25rem;
        border-radius: 16px;
        position: relative;
        max-width: 85%;
        box-shadow: var(--shadow-sm);
        font-family: 'Inter', sans-serif;
    }
    
    /* User: Radiant Blue, Right Aligned */
    .user-message {
        background-color: var(--primary);
        color: white;
        margin-left: auto; /* Push to right */
        border-bottom-right-radius: 4px;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25);
    }
    
    /* Assistant: White, Left Aligned */
    .assistant-message {
        background-color: var(--bg-main);
        color: var(--text-main);
        border: 1px solid var(--border-light);
        border-bottom-left-radius: 4px;
        box-shadow: var(--shadow-sm);
    }

    /* Citations / Sources */
    .source-card {
        background-color: var(--bg-main);
        border: 1px solid var(--border-light);
        border-radius: 10px;
        padding: 1rem;
        margin-top: 0.75rem;
        transition: all 0.2s ease;
    }
    
    .source-card:hover {
        border-color: var(--primary);
        box-shadow: var(--shadow-md);
        transform: translateY(-1px);
    }
    
    .source-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
        font-size: 0.8rem;
    }
    
    .source-badge {
        background-color: var(--bg-tertiary);
        color: var(--text-sub);
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        border: 1px solid var(--border-light);
    }
    
    .source-text {
        font-size: 0.9rem;
        color: var(--text-sub);
        line-height: 1.5;
        border-left: 2px solid var(--border-medium);
        padding-left: 0.75rem;
    }

    /* Premium Pill-Style Chat Input */
    [data-testid="stChatInput"] {
        padding: 0 !important;
        background: transparent !important;
    }

    [data-testid="stChatInput"] > div {
        background-color: #f8fafc !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 28px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
        padding: 4px 12px !important;
        margin-bottom: 2rem !important;
        transition: all 0.2s ease;
    }

    [data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #1e293b !important;
        font-size: 0.95rem !important;
        padding-top: 12px !important;
    }

    [data-testid="stChatInput"] > div:focus-within {
        background-color: #ffffff !important;
        border-color: #3b82f6 !important;
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.1) !important;
    }

    [data-testid="stChatInput"] button {
        background-color: transparent !important;
        color: #3b82f6 !important;
        margin-bottom: 4px !important;
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Multiselect Customization Hack */
    .stMultiSelect div[data-baseweb="select"] {
        border-radius: 8px !important;
        border-color: var(--border-light) !important;
        background-color: var(--bg-main) !important;
    }
    
    .stMultiSelect span[data-baseweb="tag"] {
        background-color: var(--bg-tertiary) !important;
        border: 1px solid var(--border-medium) !important;
        color: var(--text-main) !important; /* Force dark text for visibility */
    }
    
    .stMultiSelect span[data-baseweb="tag"] span {
        color: var(--text-main) !important;
    }
    </style>
""", unsafe_allow_html=True)

def render_message(role, content, citations=None):
    """Custom HTML rendering for chat messages with premium styling."""
    if role == "user":
        # Wrap user message in a right-aligned container
        st.markdown(f"""
            <div style="display: flex; justify-content: flex-end; width: 100%; margin-bottom: 1rem;">
                <div class="message-bubble user-message">
                    {content}
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Assistant message with optional citations
        st.markdown(f"""
            <div style="display: flex; flex-direction: column; width: 100%; margin-bottom: 1rem;">
                <div class="message-bubble assistant-message">
                    {content}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        if citations:
            with st.expander("📚 Verified Sources", expanded=False):
                for i, citation in enumerate(citations, 1):
                    file_name = citation.get('file_name', 'Unknown')
                    page = citation.get('page')
                    text = citation.get('text', 'No text available')
                    
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-header">
                            <span style="font-weight: 600; color: var(--primary);">Source {i}: {file_name}</span>
                            <span class="source-badge">{f"Page {page}" if page else "Ref"}</span>
                        </div>
                        <div class="source-text">{text}</div>
                    </div>
                    """, unsafe_allow_html=True)

# Initialize session state
if "chatbot" not in st.session_state:
    # Get API keys from environment
    google_api_key = os.getenv("GOOGLE_API_KEY")
    llama_api_key = os.getenv("LLAMA_API_KEY")
    
    if not google_api_key:
        st.error("⚠️ Please set GOOGLE_API_KEY environment variable or enter it in the sidebar")
    
    if google_api_key:
        try:
            st.session_state.chatbot = SyncDocumentChatbot(
                persist_dir="./storage",
                google_api_key=google_api_key,
                llama_api_key=llama_api_key,
                project_name=os.getenv("LLAMA_CLOUD_PROJECT", "Default"),
                organization_id=os.getenv("LLAMA_CLOUD_ORG_ID"),
                timeout=600  # Longer timeout for web interface
            )
            st.session_state.initialized = True
        except Exception as e:
            st.error(f"Error initializing chatbot: {e}")
            st.session_state.initialized = False
    else:
        st.session_state.initialized = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents_uploaded" not in st.session_state:
    st.session_state.documents_uploaded = False

def process_uploads(uploaded_files):
    """Helper to process and index uploaded documents."""
    if not uploaded_files:
        return
        
    with st.spinner("Uploading and indexing documents..."):
        try:
            # Save uploaded files temporarily
            temp_dir = tempfile.mkdtemp()
            file_paths = []
            
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(file_path)
            
            # Upload each file
            uploaded_count = 0
            for i, file_path in enumerate(file_paths):
                file_name = uploaded_files[i].name
                with st.spinner(f"Processing {file_name}..."):
                    try:
                        result = st.session_state.chatbot.upload_document(file_path)
                        st.success(f"✓ {file_name}: {result}")
                        uploaded_count += 1
                    except Exception as e:
                        st.error(f"❌ {file_name}: {str(e)}")
            
            if uploaded_count > 0:
                st.balloons()
            
            st.session_state.documents_uploaded = True
            # Automatically refresh file list if possible
            st.rerun()
        except Exception as e:
            st.error(f"Error uploading documents: {e}")

# Sidebar
with st.sidebar:
    # Branding removed as requested
    
    # Settings expander to keep sidebar clean
    with st.expander("⚙️ Settings", expanded=False):
        # Google API Key input (required for Gemini)
        google_api_key_input = st.text_input(
            "Google API Key (Required)",
            type="password",
            value=os.getenv("GOOGLE_API_KEY", ""),
            help="Enter your Google API key for Gemini 2.5 Flash. Get it from https://aistudio.google.com/"
        )
        
        # LlamaCloud API Key input (for LlamaParse and Managed RAG)
        llama_api_key_input = st.text_input(
            "LlamaCloud API Key (Required)",
            type="password",
            value=os.getenv("LLAMA_API_KEY", ""),
            help="Enter your LlamaCloud API key. Get it from https://cloud.llamaindex.ai/"
        )
        
        # Project Name input
        project_name_input = st.text_input(
            "LlamaCloud Project Name",
            value=os.getenv("LLAMA_CLOUD_PROJECT", "Default"),
            help="Enter your LlamaCloud project name (default is 'Default')"
        )
        
        # Org ID input
        org_id_input = st.text_input(
            "LlamaCloud Organization ID (Optional)",
            value=os.getenv("LLAMA_CLOUD_ORG_ID", ""),
            help="Enter your LlamaCloud organization ID if required"
        )
        
        if st.button("🔑 Update API Keys & Config"):
            if google_api_key_input:
                os.environ["GOOGLE_API_KEY"] = google_api_key_input
            if llama_api_key_input:
                os.environ["LLAMA_API_KEY"] = llama_api_key_input
            
            if google_api_key_input and llama_api_key_input:
                try:
                    st.session_state.chatbot = SyncDocumentChatbot(
                        persist_dir="./storage",
                        google_api_key=google_api_key_input,
                        llama_api_key=llama_api_key_input,
                        project_name=project_name_input,
                        organization_id=org_id_input if org_id_input else None,
                        timeout=600
                    )
                    st.session_state.initialized = True
                    st.success("✓ Config updated successfully!")
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.session_state.initialized = False
            else:
                st.error("Google and LlamaCloud API keys are required!")
    
    st.markdown('<div style="height: 1px; background: var(--border-light); margin: 1.5rem 0;"></div>', unsafe_allow_html=True)

    # Document Context Section
    st.markdown('<div class="sidebar-header">Document Context</div>', unsafe_allow_html=True)
    
    if st.session_state.initialized:
        try:
            available_files = st.session_state.chatbot.list_files()
            if available_files:
                # Custom "Found Papers" Widget
                st.markdown(f"""
                    <div class="sidebar-widget">
                        <div style="font-size: 0.85rem; font-weight: 500; color: var(--text-sub); margin-bottom: 0.5rem;">
                            Indexed Documents
                        </div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: var(--text-main);">
                            {len(available_files)}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                file_options = {f["name"]: f["id"] for f in available_files}
                
                # Initialize selected_file_ids if not set
                if "selected_file_ids" not in st.session_state:
                    st.session_state.selected_file_ids = []
                
                # Get currently selected file names
                current_selected_names = [
                    name for name, file_id in file_options.items() 
                    if file_id in st.session_state.selected_file_ids
                ]
                
                selected_file_names = st.multiselect(
                    "Selected Context",
                    options=list(file_options.keys()),
                    default=current_selected_names,
                    help="Only selected papers will be used as context.",
                    label_visibility="collapsed",
                    placeholder="Select papers to chat with..."
                )
                
                # Update session state
                st.session_state.selected_file_ids = [file_options[name] for name in selected_file_names]
                
                if not selected_file_names:
                     st.caption("Using all available documents")
            else:
                st.info("Top index empty. Upload documents to start.")
                st.session_state.selected_file_ids = []
        except Exception as e:
            st.warning(f"Could not fetch file list: {e}")
            st.session_state.selected_file_ids = []
    
    if st.button("Refresh List", use_container_width=True):
        if "selected_file_ids" in st.session_state:
            del st.session_state.selected_file_ids
        st.rerun()
    
    st.markdown('<div style="height: 1px; background: var(--border-light); margin: 1.5rem 0;"></div>', unsafe_allow_html=True)
    
    # System Status Section
    st.markdown('<div class="sidebar-header">System Status</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-widget">', unsafe_allow_html=True)
    if st.session_state.initialized:
        st.markdown("""
            <div class="status-indicator">
                <div class="status-dot active"></div>
                <span>System Ready</span>
            </div>
            <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.5rem; padding-left: 1rem;">
                Managed RAG • LlamaCloud
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="status-indicator">
                <div class="status-dot inactive"></div>
                <span>Not Initialized</span>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # About Section
    with st.expander("About this App", expanded=False):
        st.markdown("""
        <div style="font-size: 0.85rem; color: var(--text-sub);">
            <strong>Research Intelligence</strong> is powered by LlamaCloud and Gemini 2.5 Flash.
            <br><br>
            Features:
            <ul style="padding-left: 1rem; margin-top: 0.5rem;">
                <li>Managed Vector Index</li>
                <li>LlamaParse Technology</li>
                <li>Multi-turn Reasoning</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Top navigation / actions bar
if st.session_state.initialized:
    col_title, col_actions = st.columns([2, 1])
    with col_title:
        pass # Branding removed
    with col_actions:
        with st.popover("📤 **Upload**", use_container_width=True):
            st.markdown("#### Add Documents")
            new_files = st.file_uploader(
                "Select documents to index",
                type=["pdf", "txt", "md", "docx", "pptx", "xlsx"],
                accept_multiple_files=True,
                key="top_bar_uploader",
                label_visibility="collapsed"
            )
            if st.button("🚀 Index Documents", key="top_bar_btn", use_container_width=True):
                process_uploads(new_files)
    st.markdown('<div style="height: 1px; background: var(--border-glass); margin: 1rem 0 2rem 0;"></div>', unsafe_allow_html=True)

# Check if chatbot is initialized
if not st.session_state.initialized:
    st.warning("⚠️ Please set your Google API key in the sidebar to continue.")
    st.info("💡 Get your API key from: https://aistudio.google.com/")
    st.stop()

# Chat interface
# Helper: Run research with streaming events
def run_research_stream(question: str, conversation_id: str, file_ids: list):
    """Run the research workflow and stream events to a queue."""
    event_queue = queue.Queue()
    
    # Ensure runner is started
    if not _runner._started:
        _runner.start()
    
    # Access the chatbot instance from the main thread before context switching
    doc_chatbot = st.session_state.chatbot.chatbot

    async def _execute_flow():
        # Start the flow using the captured instance
        handler = await doc_chatbot.run_chat_flow(
            message=question,
            conversation_id=conversation_id,
            file_ids=file_ids
        )
        
        # Stream events
        async for event in handler.stream_events():
            if hasattr(event, "msg"):
                event_queue.put(event.msg)
        
        # Get final result
        result = await handler
        return result

    # Run in background thread
    future = asyncio.run_coroutine_threadsafe(_execute_flow(), _runner._loop)
    
    return event_queue, future

if st.session_state.documents_uploaded or st.session_state.get("selected_file_ids"):
    # Display chat messages
    if st.session_state.messages:
        for message in st.session_state.messages:
            render_message(message["role"], message["content"], message.get("citations"))
    else:
        # Fill the "Void" with helpful starter questions
        st.markdown(f"""
            <div style="text-align: center; margin-top: 3rem; margin-bottom: 2rem;">
                <h3 style="font-size: 1.5rem; font-weight: 600; color: var(--text-main); margin-bottom: 0.5rem;">
                    Ready to research
                </h3>
                <p style="color: var(--text-sub);">Select a common question below or type your own.</p>
            </div>
        """, unsafe_allow_html=True)
        
        cols = st.columns(2)
        suggestions = [
            "Summarize the key findings of these papers.",
            "What are the main methodologies used?",
            "Identify any conflicting results between these documents.",
            "What future research directions are suggested?"
        ]
        
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(f"💡 {suggestion}", key=f"suggest_{i}", use_container_width=True):
                    # We'll set the prompt and rerun to trigger the chat
                    st.session_state.temp_prompt = suggestion
                    st.rerun()

    # Chat input
    # Chat input
    # Always render the input widget to ensure it stays visible
    user_input = st.chat_input("Ask a question about your documents...")
    
    # Check if we have a suggested question pending
    if st.session_state.get("temp_prompt"):
        prompt = st.session_state.pop("temp_prompt")
    else:
        prompt = user_input

    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        render_message("user", prompt)
        
        # Generate response
        with st.spinner("Initializing Research Agent..."):
            try:
                # Use multi-turn chat with optional file filtering
                file_ids = st.session_state.get("selected_file_ids", [])
                conversation_id = st.session_state.get("conversation_id")
                
                # Check for suggested question triggers
                if prompt.startswith("💡 "):
                    prompt = prompt.replace("💡 ", "")
                
                # Start streaming
                event_queue, future = run_research_stream(prompt, conversation_id, file_ids)
                
                # Dynamic Status Widget
                with st.status("🧠 **Researching...**", expanded=True) as status:
                    st.write("Planning research steps...")
                    
                    # Consume events
                    while not future.done():
                        try:
                            msg = event_queue.get(timeout=0.1)
                            st.write(f"• {msg}")
                            # Update label to show latest activity
                            if len(msg) < 50:
                                status.update(label=f"🧠 **Researching**: {msg}")
                        except queue.Empty:
                            continue
                            
                    # Get any remaining events
                    while not event_queue.empty():
                         msg = event_queue.get_nowait()
                         st.write(f"• {msg}")
                         
                    status.update(label="✅ **Research Complete!**", state="complete", expanded=False)
                
                # Get final result
                result = future.result()
                
                response = result.response
                citations = result.citations
                new_conv_id = result.conversation_id
                
                # Update conversation ID if changed
                if new_conv_id:
                     st.session_state.conversation_id = new_conv_id
                
                # Render assistant message
                render_message("assistant", response, citations)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "citations": citations
                })
                
                # Rerun to cleanly display everything
                # st.rerun() # Optional: rerunning clears the ephemeral status but keeps the message
                
            except Exception as e:
                st.error(f"Error: {e}")
                # Log full error for debugging
                import traceback
                print(traceback.format_exc())
                    
else:
    # Simplified modern landing page
    st.markdown("<div style='margin-bottom: 4rem;'></div>", unsafe_allow_html=True)
    
    # Hero Title
    st.markdown(f"""
    <div style="text-align: center; width: 100%; margin-bottom: 3rem;">
        <h1 style="font-size: 3rem; font-weight: 700; color: var(--text-main); margin-bottom: 1rem; letter-spacing: -0.03em;">
            Research Intelligence
        </h1>
        <p style="font-size: 1.125rem; color: var(--text-sub); max-width: 600px; margin: 0 auto; line-height: 1.6;">
            Synthesize complex papers and documents into actionable insights using AI-powered semantic search.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Clean Card Uploader Container
    col_l, col_m, col_r = st.columns([1, 6, 1])
    with col_m:
        st.markdown("""
        <div style="
            background: var(--bg-main);
            border: 1px solid var(--border-light);
            border-radius: 16px;
            padding: 3rem 2rem;
            text-align: center;
            box-shadow: var(--shadow-xl);
        ">
            <div style="font-size: 3.5rem; margin-bottom: 1.5rem;">🧬</div>
            <h2 style="color: var(--text-main); margin-bottom: 0.75rem; font-size: 1.5rem;">Analyze your Research</h2>
            <p style="color: var(--text-sub); margin-bottom: 2rem;">Upload PDF, TXT, or MD files to start.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Pull the uploader slightly up into the card visually if possible (Streamlit limitation: can't easily overlay)
        # Instead, we just place it below with clean styling
        main_uploaded_files = st.file_uploader(
            "Upload research papers or documents",
            type=["pdf", "txt", "md", "docx", "pptx", "xlsx"],
            accept_multiple_files=True,
            key="main_uploader",
            label_visibility="collapsed"
        )
        
        if main_uploaded_files:
            st.markdown(f"<p style='text-align: center; color: var(--secondary); font-weight: 600; margin-top: 1rem;'>📎 {len(main_uploaded_files)} file(s) ready for analysis</p>", unsafe_allow_html=True)
            if st.button("🚀 Analyze Documents", use_container_width=True):
                process_uploads(main_uploaded_files)

    st.markdown("<div style='margin-bottom: 8rem;'></div>", unsafe_allow_html=True)

# Footer
st.markdown(" ") # Minimal space at bottom

