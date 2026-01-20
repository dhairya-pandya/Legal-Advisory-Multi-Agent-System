import streamlit as st
import base64
from langchain_core.messages import HumanMessage
from agents import app as agent_app, llm

st.set_page_config(page_title="Justitia AI", page_icon="⚖️")
st.title("⚖️ Justitia: AI Legal Co-Counsel")

# --- SESSION STATE ---
if "messages" not in st.session_state: st.session_state["messages"] = []
if "last_original_response" not in st.session_state: st.session_state["last_original_response"] = None

# --- SIDEBAR (Voice & Tools) ---
with st.sidebar:
    st.header("🎙️ Voice Counsel")
    # Placing audio input here keeps the main chat clean
    audio_input = st.audio_input("Tap to Speak", key="voice_rec")
    
    st.divider()
    st.header("📂 Evidence Locker")
    uploaded_file = st.file_uploader("Upload Documents", type=["png", "jpg", "pdf", "docx"])
    
    st.divider()
    st.info("Agent Activity Log")

# --- MAIN CHAT HISTORY ---
# We create a container for history so it doesn't overlap with the bottom input
chat_container = st.container()
with chat_container:
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# --- INPUT HANDLING ---
final_input = None

# 1. Handle Voice Input (Priority)
if audio_input:
    with st.spinner("🎧 Transcribing voice..."):
        try:
            b64_audio = base64.b64encode(audio_input.getvalue()).decode("utf-8")
            msg = HumanMessage(content=[
                {"type": "text", "text": "Transcribe this audio exactly. Output ONLY the text."}, 
                {"type": "media", "mime_type": "audio/wav", "data": b64_audio}
            ])
            res = llm.invoke([msg])
            final_input = res.content
        except Exception as e:
            st.error(f"Transcription Error: {e}")

# 2. Handle Text Input
# This stays fixed at the bottom
text_input = st.chat_input("Type your legal question here...")
if text_input:
    final_input = text_input

# --- PROCESS INPUT (Agent Logic) ---
if final_input:
    st.session_state["messages"].append({"role": "user", "content": final_input})
    with st.chat_message("user"):
        st.write(final_input)

    # Prepare Inputs
    chat_history = [f"{m['role']}: {m['content']}" for m in st.session_state["messages"]]
    inputs = {"messages": chat_history}
    
    if uploaded_file:
        inputs["file_data"] = {
            "name": uploaded_file.name, 
            "type": uploaded_file.type, 
            "bytes": uploaded_file.getvalue()
        }
        st.sidebar.success(f"📎 Attached: {uploaded_file.name}")

    # Run Agent
    with st.spinner("Justitia is thinking..."):
        final_res = ""
        try:
            for output in agent_app.stream(inputs):
                for name, state in output.items():
                    # Traceability in Sidebar
                    with st.sidebar.expander(f"🔹 Active: {name}", expanded=True):
                        st.write("Processing...")
                        if "context_data" in state: 
                            st.caption(str(state["context_data"])[:100]+"...")
                    
                    if "messages" in state: 
                        final_res = state["messages"][-1]
            
            st.session_state["last_original_response"] = final_res
            st.session_state["messages"].append({"role": "assistant", "content": final_res})
            st.rerun()
            
        except Exception as e:
            st.error(f"Error: {e}")

# --- POST-PROCESSING (Translation) ---
if st.session_state["last_original_response"]:
    # This container sits just above the chat input
    with st.container():
        st.markdown("---")
        c1, c2 = st.columns([1, 4])
        with c1:
            if st.button("🌐 Translate"):
                st.session_state["show_translate"] = not st.session_state.get("show_translate", False)
        
        if st.session_state.get("show_translate"):
            lang = st.selectbox("Select Language", ["Hindi", "Tamil", "Marathi", "Telugu"], key="lang_opt")
            if lang:
                with st.spinner(f"Translating to {lang}..."):
                    prompt = f"Translate to {lang}: {st.session_state['last_original_response']}"
                    trans = llm.invoke(prompt).content
                    st.info(f"**Translation ({lang}):**\n\n{trans}")