import streamlit as st
import base64
from langchain_core.messages import HumanMessage
from agents import app as agent_app, llm

st.set_page_config(page_title="Justitia AI", page_icon="⚖️")
st.title("⚖️ Justitia: AI Legal Co-Counsel")

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state: st.session_state["messages"] = []
if "last_original_response" not in st.session_state: st.session_state["last_original_response"] = None

# --- SIDEBAR: INPUTS & TOOLS ---
with st.sidebar:
    st.header("🎙️ Voice Counsel")
    # Voice Input in Sidebar (Clean UX)
    audio_input = st.audio_input("Tap to Speak", key="voice_rec")
    
    st.divider()
    
    st.header("📂 Evidence Locker")
    # Multimodal Uploader (Video, Audio, Images, Docs)
    uploaded_file = st.file_uploader(
        "Upload Evidence", 
        type=["png", "jpg", "jpeg", "pdf", "docx", "mp3", "wav", "mp4", "mov", "avi"]
    )
    
    # Media Preview
    if uploaded_file:
        ft = uploaded_file.type
        if "audio" in ft: st.audio(uploaded_file)
        elif "video" in ft: st.video(uploaded_file)
        elif "image" in ft: st.image(uploaded_file, caption="Evidence Preview")
            
    st.divider()
    st.info("Agent Activity Log")

# --- MAIN CHAT HISTORY ---
# Container ensures chat history stays above the bottom input bar
chat_container = st.container()
with chat_container:
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# --- INPUT HANDLING LOGIC ---
final_input = None

# 1. Voice Transcription (Priority)
if audio_input:
    with st.spinner("🎧 Transcribing voice..."):
        try:
            b64_audio = base64.b64encode(audio_input.getvalue()).decode("utf-8")
            # Using Gemini to transcribe accurately
            msg = HumanMessage(content=[
                {"type": "text", "text": "Transcribe this audio exactly. Output ONLY the text."}, 
                {"type": "media", "mime_type": "audio/wav", "data": b64_audio}
            ])
            res = llm.invoke([msg])
            final_input = res.content
        except Exception as e:
            st.error(f"Transcription Error: {e}")

# 2. Text Input (Fallback)
text_input = st.chat_input("Type your legal question here...")
if text_input:
    final_input = text_input

# --- AGENT EXECUTION LOOP ---
if final_input:
    # 1. Display User Message
    st.session_state["messages"].append({"role": "user", "content": final_input})
    with st.chat_message("user"):
        st.write(final_input)

    # 2. Prepare Payload
    chat_history = [f"{m['role']}: {m['content']}" for m in st.session_state["messages"]]
    inputs = {"messages": chat_history}
    
    if uploaded_file:
        inputs["file_data"] = {
            "name": uploaded_file.name, 
            "type": uploaded_file.type, 
            "bytes": uploaded_file.getvalue()
        }
        st.sidebar.success(f"📎 Processing: {uploaded_file.name}")

    # 3. Run Agent System
    with st.spinner("Justitia is analyzing evidence & laws..."):
        final_res = ""
        try:
            for output in agent_app.stream(inputs):
                for name, state in output.items():
                    # Traceability
                    with st.sidebar.expander(f"🔹 Active: {name}", expanded=True):
                        st.write("Processing...")
                        if "context_data" in state: 
                            st.caption(str(state["context_data"])[:150]+"...")
                    
                    if "messages" in state: 
                        final_res = state["messages"][-1]
            
            # 4. Cache Result & Display
            st.session_state["last_original_response"] = final_res
            st.session_state["messages"].append({"role": "assistant", "content": final_res})
            st.rerun()
            
        except Exception as e:
            st.error(f"System Error: {e}")

# --- SEPARATE POST-PROCESSING LAYER (Translation) ---
# This runs independently of the Agent Loop
if st.session_state["last_original_response"]:
    with st.container():
        st.markdown("---")
        c1, c2 = st.columns([1, 4])
        
        with c1:
            # Toggle Translation Menu
            if st.button("🌐 Translate"):
                st.session_state["show_translate"] = not st.session_state.get("show_translate", False)
        
        # Translation Logic (Zero-Latency for Agent)
        if st.session_state.get("show_translate"):
            lang = st.selectbox("Select Language", ["Hindi", "Tamil", "Marathi", "Telugu", "Kannada", "Bengali"], key="lang_opt")
            
            if lang:
                with st.spinner(f"Translating to {lang}..."):
                    try:
                        prompt = f"Translate the following legal text to {lang}. Keep legal terms accurate.\n\n{st.session_state['last_original_response']}"
                        trans = llm.invoke(prompt).content
                        st.info(f"**Translation ({lang}):**\n\n{trans}")
                    except Exception as e:
                        st.error("Translation service unavailable.")