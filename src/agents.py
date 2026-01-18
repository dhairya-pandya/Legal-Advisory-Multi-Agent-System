import os
import sys
import streamlit as st
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder # <--- NEW IMPORT

# --- 1. CONFIG SETUP ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from config import GOOGLE_API_KEY, QDRANT_URL, QDRANT_API_KEY, EMBEDDING_MODEL
except ImportError:
    from src.config import GOOGLE_API_KEY, QDRANT_URL, QDRANT_API_KEY, EMBEDDING_MODEL

# --- 2. CACHED TOOLS ---
@st.cache_resource
def get_ai_tools():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
    
    # Qdrant Client
    client = QdrantClient(
        url=QDRANT_URL, 
        api_key=QDRANT_API_KEY, 
        prefer_grpc=False,
        timeout=60
    )
    
    # Retriever Model (Fast, finds candidates)
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    
    # Reranker Model (Slow but Smart, sorts candidates)
    # This model is specifically trained to say "Yes/No" to query-document pairs
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2') 
    
    return llm, client, encoder, reranker

llm, client, encoder, reranker = get_ai_tools()

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
    elif any(x in last_msg for x in ["law", "section", "act", "punishment", "ipc", "rights", "crime", "police"]):
        return {"current_agent": "legal_clerk"}
    else:
        return {"current_agent": "senior_counsel"}

def legal_clerk(state: AgentState):
    """
    Agent B: High-Precision Legal Retrieval (Reranked).
    """
    query = state['messages'][-1]
    
    try:
        # 1. RETRIEVE (Get a wide net of candidates)
        query_vector = encoder.encode(query).tolist()
        
        # We ask for top 15 results (Recall Phase)
        hits = client.query_points(
            collection_name="legal_knowledge",
            query=query_vector,
            using="dense",
            limit=15, 
            with_payload=True
        ).points
        
        if not hits:
            return {"context_data": "No matches found.", "messages": ["No legal data found."]}

        # 2. RERANK (Precision Phase)
        # Prepare pairs: [("Query", "Document 1"), ("Query", "Document 2")...]
        docs = [hit.payload.get('full_text', '') for hit in hits]
        pairs = [[query, doc] for doc in docs]
        
        # Score pairs (0 to 1 score)
        scores = reranker.predict(pairs)
        
        # 3. FILTER & SORT
        # Combine [ (Score, Hit) ]
        ranked_hits = sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)
        
        # Keep only High Confidence results (Score > 0.05) and top 5
        final_results = []
        for score, hit in ranked_hits[:5]:
            if score > 0.05: # Threshold: If it's trash, ignore it
                text = hit.payload.get('full_text', '')
                final_results.append(f"- [Relevance: {score:.2f}] {text}")
        
        if not final_results:
             return {"context_data": "Found laws but they were irrelevant to your specific query.", "messages": ["I searched the database but the results were not relevant enough."]}
            
        return {"context_data": "\n".join(final_results), "messages": [f"I found {len(final_results)} highly relevant legal precedents."]}

    except Exception as e:
        return {"context_data": f"Database Error: {e}", "messages": ["I am having trouble accessing the legal archives."]}

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
    2. Cite the specific Acts and Sections mentioned.
    3. If evidence is missing, state general principles but warn the user.
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