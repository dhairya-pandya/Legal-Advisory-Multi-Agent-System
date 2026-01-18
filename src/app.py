import streamlit as st
from agents import app as agent_app

st.set_page_config(page_title="Justitia AI", page_icon="⚖️")

st.title("⚖️ Justitia: AI Legal Co-Counsel")
st.markdown("### Democratizing Access to Justice in India")

# Sidebar for Traceability (Hackathon Requirement)
st.sidebar.header("⚙️ Agent Activity Log")
st.sidebar.info("This panel visualizes the Multi-Agent decision path.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display Chat
for msg in st.session_state["messages"]:
    role = "user" if msg["role"] == "user" else "assistant"
    st.chat_message(role).write(msg["content"])

# Input
user_input = st.chat_input("Describe your legal issue (e.g., 'My landlord kept my deposit')...")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    inputs = {"messages": [user_input]}
    
    with st.spinner("Justitia Agents are collaborating..."):
        final_response = ""
        # Stream the graph execution
        for output in agent_app.stream(inputs):
            for agent_name, agent_state in output.items():
                # TRACEABILITY: Show which agent is working in the sidebar
                with st.sidebar.expander(f"🔹 Active: {agent_name}", expanded=True):
                    st.write(f"**State:** Processing...")
                    if "context_data" in agent_state:
                        st.json({"retrieved_context": agent_state["context_data"]})
                
                if "messages" in agent_state:
                    final_response = agent_state["messages"][-1]

    st.session_state["messages"].append({"role": "assistant", "content": final_response})
    st.chat_message("assistant").write(final_response)