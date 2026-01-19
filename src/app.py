import streamlit as st
import base64
from agents import app as agent_app

st.set_page_config(page_title="Justitia AI", page_icon="⚖️")

st.title("⚖️ Justitia: AI Legal Co-Counsel")
st.markdown("### Democratizing Access to Justice in India")

# Sidebar for Traceability
st.sidebar.header("⚙️ Agent Activity Log")
st.sidebar.info("This panel visualizes the Multi-Agent decision path.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# --- FILE UPLOADER ---
uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Evidence (Contract, Notice, Photo)", 
    type=["png", "jpg", "jpeg", "pdf", "docx"]
)

# Display Chat History
for msg in st.session_state["messages"]:
    role = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.write(msg["content"])

# Input Area
user_input = st.chat_input("Describe your legal issue (e.g., 'Check this contract for errors')...")

if user_input:
    # 1. Handle User Input
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # --- THE MEMORY FIX ---
    # Convert session state (JSON) into a list of strings for the Agent
    # Format: ["User: Hi", "AI: Hello", "User: My landlord..."]
    chat_history = []
    for msg in st.session_state["messages"]:
        role = "User" if msg["role"] == "user" else "AI"
        chat_history.append(f"{role}: {msg['content']}")

    # 2. Prepare Inputs (Sending FULL History now)
    inputs = {"messages": chat_history}
    
    # 3. Handle File Upload
    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        file_type = uploaded_file.type
        file_name = uploaded_file.name
    with st.spinner("Justitia Agents are collaborating..."):
        final_response = ""
        try:
            for output in agent_app.stream(inputs):
                for agent_name, agent_state in output.items():
                    # Traceability in Sidebar
                    with st.sidebar.expander(f"🔹 Active: {agent_name}", expanded=True):
                        st.write(f"**State:** Processing...")
                        if "context_data" in agent_state:
                            preview = agent_state["context_data"][:300] + "..." if len(agent_state["context_data"]) > 300 else agent_state["context_data"]
                            st.info(f"**Insight:** {preview}")
                    
                    if "messages" in agent_state:
                        final_response = agent_state["messages"][-1]
        
        except Exception as e:
            final_response = f"⚠️ System Error: {str(e)}"

    # 5. Display Output
    st.session_state["messages"].append({"role": "assistant", "content": final_response})
    with st.chat_message("assistant"):
        st.write(final_response)