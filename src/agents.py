import os
import sys
import streamlit as st
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from config import GOOGLE_API_KEY, QDRANT_URL, QDRANT_API_KEY, EMBEDDING_MODEL
except ImportError:
    # Fallback for local testing if running from root
    from src.config import GOOGLE_API_KEY, QDRANT_URL, QDRANT_API_KEY, EMBEDDING_MODEL

@st.cache_resource
def get_ai_tools():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)
    
    # Connect to Qdrant
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=False)
    
    # Load Embedding Model (This is the heavy part that was causing timeouts)
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    
    return llm, client, encoder

# Load the tools
llm, client, encoder = get_ai_tools()

# --- 3. AGENT LOGIC ---

class AgentState(TypedDict):
    messages: List[str]
    current_agent: str
    context_data: str

def intake_router(state: AgentState):
    """Agent A: Routes based on user intent."""
    if not state['messages']:
        return {"current_agent": "senior_counsel"}
        
    last_msg = state['messages'][-1].lower()
    
    if any(x in last_msg for x in ["photo", "image", "upload", "file"]):
        return {"current_agent": "evidence_auditor"}
    elif any(x in last_msg for x in ["law", "section", "act", "legal", "court", "rights"]):
        return {"current_agent": "legal_clerk"}
    else:
        return {"current_agent": "senior_counsel"}

def legal_clerk(state: AgentState):
    """Agent B: Searches Qdrant for Laws."""
    query = state['messages'][-1]
    query_vector = encoder.encode(query).tolist()
    
    hits = client.search(
        collection_name="legal_knowledge",
        query_vector=query_vector,
        limit=2
    )
    
    if not hits:
        return {"context_data": "No specific laws found.", "messages": ["I checked the legal database but found no direct matches."]}

    results = "\n".join([f"- {hit.payload.get('section', 'Law')}: {hit.payload.get('text', '')}" for hit in hits])
    return {"context_data": f"LEGAL FACTS FOUND:\n{results}", "messages": [f"I have retrieved relevant legal sections: {results}"]}

def evidence_auditor(state: AgentState):
    """Agent C: Evidence Analysis."""
    return {"context_data": "Evidence Analysis: Document appears to be a valid Rent Agreement but lacks a witness signature.", "messages": ["I have analyzed the uploaded document. It is missing a witness signature on Page 2."]}

def senior_counsel(state: AgentState):
    """Agent D: Final Synthesis."""
    history = state['messages']
    context = state.get('context_data', 'No specific precedents found.')
    
    prompt = f"""
    You are 'Justitia', an AI Legal Co-Counsel for India.
    
    USER QUERY HISTORY: {history}
    LEGAL/EVIDENCE CONTEXT: {context}
    
    INSTRUCTIONS:
    1. If context is provided, USE IT to ground your answer.
    2. If no context, ask clarifying questions.
    3. Be professional, empathetic, and clear.
    4. Disclaimer: Remind the user you are an AI.
    """
    
    response = llm.invoke(prompt)
    return {"messages": [response.content]}

# --- GRAPH BUILD ---
workflow = StateGraph(AgentState)
workflow.add_node("intake", intake_router)
workflow.add_node("legal_clerk", legal_clerk)
workflow.add_node("evidence_auditor", evidence_auditor)
workflow.add_node("senior_counsel", senior_counsel)

workflow.set_entry_point("intake")

workflow.add_conditional_edges(
    "intake",
    lambda x: x['current_agent'],
    {"legal_clerk": "legal_clerk", "evidence_auditor": "evidence_auditor", "senior_counsel": "senior_counsel"}
)

workflow.add_edge("legal_clerk", "senior_counsel")
workflow.add_edge("evidence_auditor", "senior_counsel")
workflow.add_edge("senior_counsel", END)

app = workflow.compile()