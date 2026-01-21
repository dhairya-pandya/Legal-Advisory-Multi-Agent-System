import streamlit as st
import base64
from langchain_core.messages import HumanMessage
from agents import app as agent_app, llm

# --- 1. CONFIGURATION ---
PAGE_TITLE = "Justitia AI"
PAGE_ICON = "⚖️"
SUPPORTED_FILES = ["png", "jpg", "jpeg", "pdf", "docx", "mp3", "wav", "mp4", "mov", "avi", "mkv"]

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

# --- 2. SESSION STATE ---
def init_state():
    """Initializes session state safely."""
    defaults = {
        "messages": [],
        "last_original_response": None,
        "show_translate": False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_state()

# --- 3. UI FUNCTIONS ---

def render_sidebar():
    """Renders Sidebar Inputs and returns selected values."""
    with st.sidebar:
        st.header("🎙️ Voice Counsel")
        audio_in = st.audio_input("Tap to Speak", key="voice_rec")
        
        st.divider()
        st.header("📂 Evidence Locker")
        uploaded_file = st.file_uploader("Upload Evidence", type=SUPPORTED_FILES)
        
        if uploaded_file:
            ft = uploaded_file.type
            st.caption(f"Preview: {uploaded_file.name}")
            if "audio" in ft: st.audio(uploaded_file)
            elif "video" in ft: st.video(uploaded_file)
            elif "image" in ft: st.image(uploaded_file)
            
        st.divider()
        
        # --- PRIVACY CONTROLS ---
        st.header("🔒 Privacy")
        incognito = st.toggle("Incognito Mode", value=False, help="When active, your queries and evidence will NOT be saved to the learning cache.")
        if incognito:
            st.caption("✅ Data Retention: Disabled")
        
        st.divider()
        st.info("Agent Activity Log")
        return audio_in, uploaded_file, incognito

def process_voice_input(audio_input):
    """Transcribes voice to text."""
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

def render_chat():
    """Renders Chat History."""
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

def render_translation_tools():
    """Renders Translation Options if a response exists."""
    if st.session_state["last_original_response"]:
        st.markdown("---")
        c1, c2 = st.columns([1, 4])
        with c1:
            if st.button("🌐 Translate Output"):
                st.session_state["show_translate"] = not st.session_state["show_translate"]
        
        if st.session_state["show_translate"]:
            lang = st.selectbox("Select Language", 
                ["Hindi", "Tamil", "Marathi", "Telugu", "Kannada", "Bengali", "Gujarati", "Malayalam"]
            )
            if lang:
                with st.spinner(f"Translating to {lang}..."):
                    try:
                        prompt = f"Translate this legal advice to {lang}. Maintain accuracy.\n\n{st.session_state['last_original_response']}"
                        trans = llm.invoke(prompt).content
                        st.success(f"**Translation ({lang}):**\n\n{trans}")
                    except Exception:
                        st.error("Translation unavailable.")

# --- 4. MAIN FLOW ---

def main():
    # A. Render Layout
    c1, c2 = st.columns([1, 8])
    with c1: st.image("https://cdn-icons-png.flaticon.com/512/924/924915.png", width=60)
    with c2: st.title("Justitia: AI Legal Co-Counsel")
    
    # Capture Inputs (including Incognito flag)
    audio_val, file_val, is_incognito = render_sidebar()
    
    render_chat()
    render_translation_tools()

    # B. Input Handling
    final_input = None
    
    # Priority 1: Voice
    if audio_val:
        transcription = process_voice_input(audio_val)
        if transcription: final_input = transcription
    
    # Priority 2: Text (Only if no voice)
    if not final_input:
        text_val = st.chat_input("Describe your legal issue...")
        if text_val: final_input = text_val

    # C. Processing Logic
    if final_input:
        # 1. Update UI immediately
        st.session_state["messages"].append({"role": "user", "content": final_input})
        with st.chat_message("user"):
            st.write(final_input)

        # 2. Prepare Inputs for Backend
        chat_history = [f"{m['role']}: {m['content']}" for m in st.session_state["messages"]]
        
        # Inject the privacy flag into the state
        inputs = {
            "messages": chat_history,
            "is_incognito": is_incognito,
            "context_data": "" # <--- Simple String Initialization
        }
        
        if file_val:
            inputs["file_data"] = {
                "name": file_val.name, 
                "type": file_val.type, 
                "bytes": file_val.getvalue()
            }
            st.sidebar.success(f"📎 Analyzing: {file_val.name}")

        # 3. Stream Agents
        with st.chat_message("assistant"):
            with st.spinner("Justitia is thinking..."):
                final_res = ""
                try:
                    for output in agent_app.stream(inputs):
                        for agent_name, agent_state in output.items():
                            
                            # Traceability Log
                            with st.sidebar.expander(f"🔹 Active: {agent_name}", expanded=True):
                                st.caption("Status: Processing...")
                                # Simple String Display
                                if "context_data" in agent_state:
                                    snippet = agent_state["context_data"][-200:] # Show last 200 chars
                                    st.code(snippet, language="text")

                            # Capture Final Message
                            if "messages" in agent_state:
                                final_res = agent_state["messages"][-1]
                    
                    st.write(final_res)
                    
                    # 4. Save & Reset
                    st.session_state["messages"].append({"role": "assistant", "content": final_res})
                    st.session_state["last_original_response"] = final_res
                    st.rerun()

                except Exception as e:
                    st.error(f"System Error: {e}")

if __name__ == "__main__":
    main()