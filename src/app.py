import streamlit as st
import base64
from agents import app as agent_app

st.set_page_config(page_title="Justitia AI", page_icon="⚖️")

st.title("⚖️ Justitia: AI Legal Co-Counsel")
st.markdown("### Democratizing Access to Justice in India")

# Sidebar for Traceability (Hackathon Requirement)
st.sidebar.header("⚙️ Agent Activity Log")
st.sidebar.info("This panel visualizes the Multi-Agent decision path.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# --- FILE UPLOADER (NEW FEATURE) ---
uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Evidence (Contract, Notice, Photo)", 
    type=["png", "jpg", "jpeg", "pdf", "docx"]
)

# Display Chat History
for msg in st.session_state["messages"]:
    role = "user" if msg["role"] == "user" else "assistant"
    st.chat_message(role).write(msg["content"])

# Input Area
user_input = st.chat_input("Describe your legal issue (e.g., 'Check this contract for errors')...")

if user_input:
    # 1. Handle User Input
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # 2. Prepare Inputs for Agents
    inputs = {"messages": [user_input]}
    
    # 3. Handle File Upload (Pass file data to Agents)
    if uploaded_file:
        # We convert to bytes so it can be passed through the LangGraph state
        file_bytes = uploaded_file.getvalue()
        file_type = uploaded_file.type
        file_name = uploaded_file.name
        
        inputs["file_data"] = {
            "name": file_name,
            "type": file_type,
            "bytes": file_bytes
        }
        st.sidebar.success(f"📎 Attached: {file_name}")

    # 4. Run the Agent Graph
    with st.spinner("Justitia Agents are collaborating..."):
        final_response = ""
        try:
            # Stream the graph execution
            for output in agent_app.stream(inputs):
                for agent_name, agent_state in output.items():
                    # TRACEABILITY: Show which agent is working in the sidebar
                    with st.sidebar.expander(f"🔹 Active: {agent_name}", expanded=True):
                        st.write(f"**State:** Processing...")
                        
                        # Show retrieved context or analysis in sidebar for judges
                        if "context_data" in agent_state:
                            # Truncate long text for display
                            preview = agent_state["context_data"][:300] + "..." if len(agent_state["context_data"]) > 300 else agent_state["context_data"]
                            st.info(f"**Insight:** {preview}")
                    
                    if "messages" in agent_state:
                        final_response = agent_state["messages"][-1]
        
        except Exception as e:
            final_response = f"⚠️ System Error: {str(e)}"

    # 5. Display Output
    st.session_state["messages"].append({"role": "assistant", "content": final_response})
    st.chat_message("assistant").write(final_response)