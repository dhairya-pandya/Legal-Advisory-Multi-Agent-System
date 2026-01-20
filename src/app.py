import streamlit as st
import base64
from langchain_core.messages import HumanMessage
from agents import app as agent_app, llm

# --- CONFIG & CONSTANTS ---
PAGE_TITLE = "Justitia AI"
PAGE_ICON = "⚖️"
SUPPORTED_FILES = ["png", "jpg", "jpeg", "pdf", "docx", "mp3", "wav", "mp4", "mov", "avi"]

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)
st.title(f"{PAGE_ICON} {PAGE_TITLE}: Legal Co-Counsel")

# --- 1. STATE MANAGEMENT ---
def init_state():
    """Initializes session state variables if they don't exist."""
    defaults = {
        "messages": [],
        "last_original_response": None,
        "show_translate": False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_state()

# --- 2. UI COMPONENTS ---

def render_sidebar():
    """Renders the sidebar and returns inputs."""
    with st.sidebar:
        st.header("🎙️ Voice Counsel")
        audio_input = st.audio_input("Tap to Speak", key="voice_rec")
        
        st.divider()
        st.header("📂 Evidence Locker")
        uploaded_file = st.file_uploader("Upload Evidence", type=SUPPORTED_FILES)
        
        # Preview Logic
        if uploaded_file:
            ft = uploaded_file.type
            if "audio" in ft: st.audio(uploaded_file)
            elif "video" in ft: st.video(uploaded_file)
            elif "image" in ft: st.image(uploaded_file, caption="Evidence Preview")
            
        st.divider()
        st.info("Agent Activity Log")
        
        return audio_input, uploaded_file

def render_chat_history():
    """Renders the chat messages."""
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

def process_voice_input(audio_input):
    """Transcribes voice input using LLM."""
    if not audio_input: return None
    
    with st.spinner("🎧 Transcribing voice..."):
        try:
            b64_audio = base64.b64encode(audio_input.getvalue()).decode("utf-8")
            msg = HumanMessage(content=[
                {"type": "text", "text": "Transcribe this audio exactly. Output ONLY the text."}, 
                {"type": "media", "mime_type": "audio/wav", "data": b64_audio}
            ])
            res = llm.invoke([msg])
            return res.content
        except Exception as e:
            st.error(f"Transcription Error: {e}")
            return None

def handle_agent_execution(user_input, uploaded_file):
    """Runs the LangGraph Agent Workflow."""
    # 1. Update UI with User Message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.rerun() # Quick rerun to show user message immediately

def run_agent_logic(user_input, uploaded_file):
    # This function is called AFTER the rerun to process data
    chat_history = [f"{m['role']}: {m['content']}" for m in st.session_state["messages"]]
    inputs = {"messages": chat_history}
    
    if uploaded_file:
        inputs["file_data"] = {
            "name": uploaded_file.name, 
            "type": uploaded_file.type, 
            "bytes": uploaded_file.getvalue()
        }
        st.sidebar.success(f"📎 Processing: {uploaded_file.name}")

    with st.spinner("Justitia is analyzing evidence & laws..."):
        final_res = ""
        try:
            for output in agent_app.stream(inputs):
                for name, state in output.items():
                    with st.sidebar.expander(f"🔹 Active: {name}", expanded=True):
                        st.write("Processing...")
                        if "context_data" in state: 
                            st.caption(str(state["context_data"])[:150]+"...")
                    
                    if "messages" in state: 
                        final_res = state["messages"][-1]
            
            st.session_state["last_original_response"] = final_res
            st.session_state["messages"].append({"role": "assistant", "content": final_res})
            st.rerun()
            
        except Exception as e:
            st.error(f"System Error: {e}")

# --- 3. MAIN EXECUTION FLOW ---

# A. Render UI
audio_in, upload_in = render_sidebar()
render_chat_history()

# B. Handle Inputs
final_input = None
voice_text = process_voice_input(audio_in)
text_input = st.chat_input("Type your legal question here...")

if voice_text: final_input = voice_text
elif text_input: final_input = text_input

# C. Logic Trigger
if final_input:
    # Add to state and run logic
    # Note: Streamlit execution flow is tricky. 
    # To keep it simple for this hackathon, we combine the append + run logic here.
    st.session_state["messages"].append({"role": "user", "content": final_input})
    run_agent_logic(final_input, upload_in)

# D. Post-Processing (Translation)
if st.session_state["last_original_response"]:
    with st.container():
        st.markdown("---")
        c1, c2 = st.columns([1, 4])
        
        with c1:
            if st.button("🌐 Translate"):
                st.session_state["show_translate"] = not st.session_state["show_translate"]
        
        if st.session_state["show_translate"]:
            lang = st.selectbox("Select Language", ["Hindi", "Tamil", "Marathi", "Telugu", "Kannada", "Bengali"], key="lang_opt")
            
            if lang:
                with st.spinner(f"Translating to {lang}..."):
                    try:
                        prompt = f"Translate the following legal text to {lang}. Keep legal terms accurate.\n\n{st.session_state['last_original_response']}"
                        trans = llm.invoke(prompt).content
                        st.info(f"**Translation ({lang}):**\n\n{trans}")
                    except Exception:
                        st.error("Translation service unavailable.")