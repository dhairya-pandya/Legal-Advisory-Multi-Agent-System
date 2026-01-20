import streamlit as st
import base64
from langchain_core.messages import HumanMessage
from agents import app as agent_app, llm  # Import llm for transcription

st.set_page_config(page_title="Justitia AI", page_icon="⚖️")

st.title("⚖️ Justitia: AI Legal Co-Counsel")
st.markdown("### Democratizing Access to Justice in India")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# --- SIDEBAR CONFIG ---
st.sidebar.header("⚙️ Settings")
language = st.sidebar.selectbox(
    "🗣️ Output Language",
    ["English", "Hindi", "Marathi", "Tamil", "Telugu", "Kannada", "Bengali", "Gujarati"]
)

st.sidebar.info("This panel visualizes the Multi-Agent decision path.")
uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Evidence", 
    type=["png", "jpg", "jpeg", "pdf", "docx"]
)

# --- DISPLAY CHAT ---
for msg in st.session_state["messages"]:
    role = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.write(msg["content"])

# --- INPUT HANDLERS ---
# 1. Voice Input
audio_value = st.audio_input("🎤 Record Voice Command")

# 2. Text Input
text_input = st.chat_input("Type your legal question...")

final_input = None

# LOGIC: Handle Voice OR Text
if audio_value:
    # Transcribe Audio using Gemini (Multimodal)
    with st.spinner("🎧 Transcribing voice..."):
        try:
            audio_bytes = audio_value.getvalue()
            # We treat audio as a "file" for Gemini to transcribe
            # Note: We need to pass audio bytes correctly. 
            # Since Gemini 1.5 Flash is multimodal, we can send audio blob directly.
            b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
            
            msg = HumanMessage(content=[
                {"type": "text", "text": "Transcribe this audio exactly. Do not add any other text."},
                {"type": "media", "mime_type": "audio/wav", "data": b64_audio}
            ])
            # For simplicity in this hackathon, let's assume it's English/Hindi mixed
            # Ideally, we use the 'llm' imported from agents
            res = llm.invoke([msg])
            transcribed_text = res.content
            
            st.success(f"🗣️ You said: {transcribed_text}")
            final_input = transcribed_text
        except Exception as e:
            st.error(f"Audio Error: {e}")

elif text_input:
    final_input = text_input

# --- PROCESS INPUT ---
if final_input:
    # Append User Message
    st.session_state["messages"].append({"role": "user", "content": final_input})
    with st.chat_message("user"):
        st.write(final_input)

    # Convert History to Strings
    chat_history = [f"{m['role']}: {m['content']}" for m in st.session_state["messages"]]
    
    # Prepare Inputs
    # We pass the 'language' preference to the agent state so Senior Counsel knows!
    inputs = {
        "messages": chat_history,
        "context_data": f"User Preferred Language: {language}" 
    }
    
    # Handle File
    if uploaded_file:
        inputs["file_data"] = {
            "name": uploaded_file.name,
            "type": uploaded_file.type,
            "bytes": uploaded_file.getvalue()
        }
        st.sidebar.success(f"📎 Attached: {uploaded_file.name}")

    # Run Agents
    with st.spinner(f"Justitia is analyzing in {language}..."):
        final_response = ""
        try:
            for output in agent_app.stream(inputs):
                for agent_name, agent_state in output.items():
                    with st.sidebar.expander(f"🔹 Active: {agent_name}", expanded=True):
                        st.write("Processing...")
                        # Show context if updated (handling the String format)
                        if "context_data" in agent_state:
                             # Just show a snippet
                             st.caption(str(agent_state["context_data"])[:100] + "...")

                    if "messages" in agent_state:
                        final_response = agent_state["messages"][-1]
        
        except Exception as e:
            final_response = f"⚠️ System Error: {str(e)}"

    # Display Output
    st.session_state["messages"].append({"role": "assistant", "content": final_response})
    with st.chat_message("assistant"):
        st.write(final_response)