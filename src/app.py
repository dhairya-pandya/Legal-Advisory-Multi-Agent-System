import streamlit as st
import base64
from langchain_core.messages import HumanMessage
from agents import app as agent_app, llm

# --- CONFIG ---
PAGE_TITLE = "Justitia AI"
PAGE_ICON = "⚖️"
SUPPORTED_FILES = ["png", "jpg", "jpeg", "pdf", "docx", "mp3", "wav", "mp4", "mov", "avi", "mkv"]

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

# --- STATE ---
def init_state():
    if "messages" not in st.session_state: st.session_state["messages"] = []
    if "last_original_response" not in st.session_state: st.session_state["last_original_response"] = None
    if "show_translate" not in st.session_state: st.session_state["show_translate"] = False

init_state()

# --- UI ---
def render_sidebar():
    with st.sidebar:
        st.header("🎙️ Voice Counsel")
        audio_in = st.audio_input("Tap to Speak", key="voice_rec")
        
        st.divider()
        st.header("📂 Evidence Locker")
        uploaded_file = st.file_uploader("Upload Evidence", type=SUPPORTED_FILES)
        
        if uploaded_file:
            st.caption(f"Preview: {uploaded_file.name}")
            ft = str(uploaded_file.type).lower()
            if "audio" in ft: st.audio(uploaded_file)
            elif "video" in ft: st.video(uploaded_file)
            elif "image" in ft: st.image(uploaded_file)
            
        st.divider()
        st.header("🔒 Privacy")
        incognito = st.toggle("Incognito Mode", value=False)
        if incognito: st.caption("✅ Cache Disabled")
        
        st.divider()
        st.info("Agent Activity Log")
        return audio_in, uploaded_file, incognito

def process_voice(audio_input):
    if not audio_input: return None
    with st.spinner("Transcribing..."):
        try:
            b64 = base64.b64encode(audio_input.getvalue()).decode("utf-8")
            msg = HumanMessage(content=[{"type":"text","text":"Transcribe exactly."}, {"type":"media","mime_type":"audio/wav","data":b64}])
            return llm.invoke([msg]).content
        except Exception as e:
            st.error(f"Error: {e}")
            return None

def main():
    c1, c2 = st.columns([1, 8])
    with c1: st.image("https://cdn-icons-png.flaticon.com/512/924/924915.png", width=60)
    with c2: st.title("Justitia: AI Legal Co-Counsel")
    
    audio_val, file_val, is_incognito = render_sidebar()
    
    # Chat History
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]): st.write(msg["content"])

    # Input Logic
    final_input = None
    if audio_val: final_input = process_voice(audio_val)
    if not final_input: final_input = st.chat_input("Legal Query...")

    if final_input:
        st.session_state["messages"].append({"role": "user", "content": final_input})
        with st.chat_message("user"): st.write(final_input)

        # Prepare Payload
        chat_history = [f"{m['role']}: {m['content']}" for m in st.session_state["messages"]]
        inputs = {"messages": chat_history, "is_incognito": is_incognito}
        
        if file_val:
            # SAFETY FIX: Force string type
            safe_type = str(file_val.type) if file_val.type else ""
            inputs["file_data"] = {
                "name": file_val.name, 
                "type": safe_type, 
                "bytes": file_val.getvalue()
            }
            st.sidebar.success(f"📎 Processing: {file_val.name}")

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                final_res = ""
                try:
                    for output in agent_app.stream(inputs):
                        for name, state in output.items():
                            with st.sidebar.expander(f"🔹 {name}", expanded=True):
                                if "context_data" in state: 
                                    st.code(str(state["context_data"])[:200])
                            if "messages" in state: final_res = state["messages"][-1]
                    
                    st.write(final_res)
                    st.session_state["messages"].append({"role": "assistant", "content": final_res})
                    st.session_state["last_original_response"] = final_res
                    st.rerun()
                except Exception as e:
                    st.error(f"System Error: {e}")

    # Translation
    if st.session_state["last_original_response"]:
        st.markdown("---")
        if st.button("🌐 Translate Output"):
            st.session_state["show_translate"] = not st.session_state["show_translate"]
        
        if st.session_state["show_translate"]:
            lang = st.selectbox("Language", ["Hindi", "Tamil", "Marathi", "Telugu"])
            if lang:
                with st.spinner("Translating..."):
                    p = f"Translate to {lang}: {st.session_state['last_original_response']}"
                    st.info(llm.invoke(p).content)

if __name__ == "__main__":
    main()