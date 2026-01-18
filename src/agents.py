import os
import sys
import streamlit as st
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
# We don't strictly need SparseTextEmbedding here for a basic search, 
# but we need to target the "dense" vector we created.

# --- 1. SETUP & IMPORTS ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from config import GOOGLE_API_KEY, QDRANT_URL, QDRANT_API_KEY, EMBEDDING_MODEL
except ImportError:
    from src.config import GOOGLE_API_KEY, QDRANT_URL, QDRANT_API_KEY, EMBEDDING_MODEL

# --- 2. INITIALIZE TOOLS ---
@st.cache_resource
def get_ai_tools():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
    
    # Connect to Qdrant (Increased timeout for safety)
    client = QdrantClient(
        url=QDRANT_URL, 
        api_key=QDRANT_API_KEY, 
        prefer_grpc=False,
        timeout=60
    )
    
    # Load Dense Model (Concepts)
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    
    return llm, client, encoder

llm, client, encoder = get_ai_tools()

# --- 3. AGENT DEFINITIONS ---

class AgentState(TypedDict):
    messages: List[str]
    current_agent: str
    context_data: str

def intake_router(state: AgentState):
    """Routes the user query to the right specialist."""
    if not state['messages']:
        return {"current_agent": "senior_counsel"}
        
    last_msg = state['messages'][-1].lower()
    
    # Routing Logic
    if any(x in last_msg for x in ["photo", "image", "upload", "file", "scan"]):
        return {"current_agent": "evidence_auditor"}
    elif any(x in last_msg for x in ["law", "section", "act", "legal", "court", "rights", "punishment", "ipc"]):
        return {"current_agent": "legal_clerk"}
    else:
        return {"current_agent": "senior_counsel"}

def legal_clerk(state: AgentState):
    """
    Agent B: Searches the Massive 34k Legal Database.
    FIX: Targets the 'dense' named vector specifically.
    """
    query = state['messages'][-1]
    
    # 1. Generate Vector
    query_vector = encoder.encode(query).tolist()
    
    try:
        # 2. Search Targeting the "dense" Vector (The Fix)
        # We use a Tuple ("vector_name", vector_data) to be compatible with all client versions
        hits = client.search(
            collection_name="legal_knowledge",
            query_vector=("dense", query_vector), 
            limit=5,
            with_payload=True
        )
    except Exception as e:
        return {"context_data": f"Search failed: {e}", "messages": ["I am having trouble connecting to the legal archives."]}
    
    if not hits:
        return {"context_data": "No matches found.", "messages": ["I searched the database but found no specific laws matching your query."]}

    # 3. Parse Results (Using the new 'full_text' field from your ingestion script)
    results = []
    for hit in hits:
        # Your ingest script saved it as 'full_text', so we grab that
        text = hit.payload.get('full_text', hit.payload.get('text', 'Law Text Missing'))
        results.append(f"- {text}")

    formatted_results = "\n".join(results)
    return {
        "context_data": f"LEGAL FACTS FOUND:\n{formatted_results}", 
        "messages": [f"I have found {len(hits)} relevant legal sections from the database."]
    }

def evidence_auditor(state: AgentState):
    """Agent C: Placeholder for Document Analysis."""
    return {
        "context_data": "Evidence Analysis: Document appears to be a Rent Agreement. Missing witness signature.", 
        "messages": ["I have analyzed the document. It appears valid but lacks a witness signature."]
    }

def senior_counsel(state: AgentState):
    """Agent D: Synthesizes the final answer."""
    history = state['messages']
    context = state.get('context_data', 'No context provided.')
    
    prompt = f"""
    You are 'Justitia', an AI Legal Co-Counsel for India.
    
    USER QUERY: {history}
    LEGAL EVIDENCE FROM DATABASE: {context}
    
    INSTRUCTIONS:
    1. Answer the user's legal question using ONLY the provided evidence.
    2. Cite the specific Acts and Sections mentioned (e.g., "According to IPC Section 302...").
    3. If the evidence is irrelevant, state general legal principles but warn the user.
    4. Keep it professional and empathetic.
    """
    
    response = llm.invoke(prompt)
    return {"messages": [response.content]}

# --- 4. GRAPH CONSTRUCTION ---
workflow = StateGraph(AgentState)
workflow.add_node("intake", intake_router)
workflow.add_node("legal_clerk", legal_clerk)
workflow.add_node("evidence_auditor", evidence_auditor)
workflow.add_node("senior_counsel", senior_counsel)

workflow.set_entry_point("intake")

workflow.add_conditional_edges("intake", lambda x: x['current_agent'], 
    {"legal_clerk": "legal_clerk", "evidence_auditor": "evidence_auditor", "senior_counsel": "senior_counsel"})

workflow.add_edge("legal_clerk", "senior_counsel")
workflow.add_edge("evidence_auditor", "senior_counsel")
workflow.add_edge("senior_counsel", END)

app = workflow.compile()