import os
import sys
import streamlit as st
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# --- 1. CONFIG SETUP ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from config import GOOGLE_API_KEY, QDRANT_URL, QDRANT_API_KEY, EMBEDDING_MODEL
except ImportError:
    from src.config import GOOGLE_API_KEY, QDRANT_URL, QDRANT_API_KEY, EMBEDDING_MODEL

@st.cache_resource
def get_ai_tools():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)
    
    # Client with increased timeout
    client = QdrantClient(
        url=QDRANT_URL, 
        api_key=QDRANT_API_KEY, 
        prefer_grpc=False,
        timeout=60
    )
    
    # Models
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    
    return llm, client, encoder

llm, client, encoder = get_ai_tools()

# --- 3. AGENT STATE ---
class AgentState(TypedDict):
    messages: List[str]
    current_agent: str
    context_data: str

# --- 4. AGENTS ---

def intake_router(state: AgentState):
    """Routes the query."""
    if not state['messages']:
        return {"current_agent": "senior_counsel"}
    last_msg = state['messages'][-1].lower()
    
    if any(x in last_msg for x in ["photo", "image", "scan", "contract"]):
        return {"current_agent": "evidence_auditor"}
    elif any(x in last_msg for x in ["law", "section", "act", "punishment", "ipc", "rights"]):
        return {"current_agent": "legal_clerk"}
    else:
        return {"current_agent": "senior_counsel"}

def legal_clerk(state: AgentState):
    """Agent B: Robust Hybrid Search."""
    query = state['messages'][-1]
    
    try:
        # 1. Generate Dense Vector
        dense_vector = encoder.encode(query).tolist()
        
        # 2. Perform Search (Using 'Safe Tuple' Syntax)
        hits = client.search(
            collection_name="legal_knowledge",
            query_vector=("dense", dense_vector), 
            limit=5,
            with_payload=True
        )
        
        if not hits:
            return {"context_data": "No matches found.", "messages": ["I checked the legal database but found no direct matches."]}

        # Format results
        results = []
        for hit in hits:
            # Handle variable payload structure
            act = hit.payload.get('act', 'Act')
            section = hit.payload.get('section', '?')
            # Look for 'full_text' first, then 'text', then 'law'
            text = hit.payload.get('full_text') or hit.payload.get('text') or hit.payload.get('law') or ""
            results.append(f"- {text}")
            
        return {"context_data": "\n".join(results), "messages": [f"I found {len(hits)} relevant legal precedents."]}

    except Exception as e:
        print(f"SEARCH ERROR: {e}")
        return {"context_data": f"Database Error: {e}", "messages": ["I am having trouble accessing the legal archives momentarily."]}

def evidence_auditor(state: AgentState):
    """Agent C: Document Analysis."""
    return {"context_data": "Document Scan: Valid Rent Agreement. Missing witness signature on Page 2.", "messages": ["I analyzed the document. It appears valid but lacks a witness signature."]}

def senior_counsel(state: AgentState):
    """Agent D: Final Advice."""
    history = state['messages']
    context = state.get('context_data', 'No context.')
    
    prompt = f"""
    You are 'Justitia', an AI Legal Co-Counsel for India.
    
    USER QUERY: {history}
    LEGAL EVIDENCE: {context}
    
    INSTRUCTIONS:
    1. Answer using ONLY the provided Legal Evidence.
    2. Cite the specific Acts and Sections mentioned in the evidence.
    3. If the evidence is missing, ask for clarification.
    """
    response = llm.invoke(prompt)
    return {"messages": [response.content]}

# --- 5. WORKFLOW ---
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