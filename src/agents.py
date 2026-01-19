import os
import sys
import io
import base64
import operator
import streamlit as st
from typing import TypedDict, List, Optional, Annotated
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain_core.messages import HumanMessage
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import pypdf
import docx

# --- 1. CREDENTIALS & CONFIG ---
def get_credentials():
    creds = {}
    try:
        import config
        creds["GOOGLE_API_KEY"] = getattr(config, "GOOGLE_API_KEY", None)
        creds["QDRANT_URL"] = getattr(config, "QDRANT_URL", None)
        creds["QDRANT_API_KEY"] = getattr(config, "QDRANT_API_KEY", None)
        creds["EMBEDDING_MODEL"] = getattr(config, "EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    except ImportError:
        pass
        
    if not creds.get("GOOGLE_API_KEY"):
        creds["GOOGLE_API_KEY"] = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
        creds["QDRANT_URL"] = st.secrets.get("QDRANT_URL", os.getenv("QDRANT_URL"))
        creds["QDRANT_API_KEY"] = st.secrets.get("QDRANT_API_KEY", os.getenv("QDRANT_API_KEY"))
        creds["EMBEDDING_MODEL"] = st.secrets.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    return creds

CREDS = get_credentials()

if not CREDS["GOOGLE_API_KEY"] or not CREDS["QDRANT_URL"]:
    st.error("🚨 Configuration Error: API Keys not found.")
    st.stop()

# --- 2. CACHED TOOLS ---
@st.cache_resource
def get_ai_tools():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=CREDS["GOOGLE_API_KEY"],
        temperature=0.3,
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }
    )
    client = QdrantClient(url=CREDS["QDRANT_URL"], api_key=CREDS["QDRANT_API_KEY"], prefer_grpc=False, timeout=60)
    encoder = SentenceTransformer(CREDS["EMBEDDING_MODEL"])
    return llm, client, encoder

llm, client, encoder = get_ai_tools()

# --- 3. AGENT STATE ---
class AgentState(TypedDict):
    messages: List[str]
    # operator.add merges lists. We must ensure NO agent returns None for this key.
    context_data: Annotated[List[str], operator.add] 
    file_data: Optional[dict]

# --- 4. NODES & ROUTING ---

def intake(state: AgentState):
    """Dummy Entry Node"""
    return {} 

def route_logic(state: AgentState) -> List[str]:
    """Router Logic"""
    agents_to_run = []
    
    # Check messages safely
    if state.get('messages') and len(state['messages']) > 0:
        agents_to_run.append("legal_clerk")
        
    # Check file data safely
    if state.get("file_data"):
        agents_to_run.append("evidence_auditor")

    if not agents_to_run:
        return ["senior_counsel"]
        
    return agents_to_run

# --- 5. AGENTS ---

def legal_clerk(state: AgentState):
    """Retrieves Legal Context (Standard Search)."""
    # Defensive check
    if not state.get('messages'):
        return {"context_data": []}
        
    query = state['messages'][-1].split("User: ")[-1]
    
    try:
        # 1. Search (Dense Vector)
        hits = client.query_points(
            collection_name="legal_knowledge",
            query=encoder.encode(query).tolist(),
            using="dense",
            limit=5, # Reduced limit since we removed Reranker
            with_payload=True
        ).points
        
        if not hits:
            return {"context_data": ["Legal Clerk: No specific laws found."]}

        # 2. Format Results (Directly, no Reranker)
        final_results = []
        for hit in hits:
            # Robust payload access
            text = hit.payload.get('full_text') or hit.payload.get('text') or "No Text Available"
            final_results.append(f"- {text}")
            
        combined_text = "LEGAL PRECEDENTS FOUND:\n" + "\n".join(final_results)
        
        # CRITICAL: Always return a LIST
        return {"context_data": [combined_text]}

    except Exception as e:
        # Return error as context so the graph doesn't crash
        return {"context_data": [f"Legal Clerk Error: {str(e)}"]}

def evidence_auditor(state: AgentState):
    """Analyzes Files."""
    file_data = state.get("file_data")
    if not file_data:
        return {"context_data": []}

    file_name = file_data.get("name", "Unknown")
    file_type = file_data.get("type", "")
    file_bytes = file_data.get("bytes")
    
    try:
        # IMAGE HANDLER
        if "image" in file_type or file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            b64_image = base64.b64encode(file_bytes).decode('utf-8')
            msg = HumanMessage(content=[
                {"type": "text", "text": "Forensic Analysis: 1. Transcribe text. 2. Flag issues."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
            ])
            response = llm.invoke([msg])
            return {"context_data": [f"EVIDENCE ANALYSIS ({file_name}):\n{response.content}"]}

        # PDF HANDLER
        elif "pdf" in file_type:
            pdf = pypdf.PdfReader(io.BytesIO(file_bytes))
            text = "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])
            return {"context_data": [f"FILE TEXT ({file_name}):\n{text[:5000]}"]}
            
        # DOCX HANDLER
        elif "word" in file_type:
            doc = docx.Document(io.BytesIO(file_bytes))
            text = "\n".join([p.text for p in doc.paragraphs])
            return {"context_data": [f"FILE TEXT ({file_name}):\n{text[:5000]}"]}
            
        return {"context_data": ["Evidence Auditor: Unsupported file type."]}

    except Exception as e:
        return {"context_data": [f"Evidence Auditor Error: {str(e)}"]}

def senior_counsel(state: AgentState):
    """Synthesizes Final Answer."""
    history = state.get('messages', [])
    
    # Safe access to context_data
    context_list = state.get('context_data', [])
    # Ensure it's a list before joining
    if context_list is None: context_list = []
    
    full_evidence = "\n\n".join([str(c) for c in context_list]) if context_list else "No evidence provided."
    
    prompt = f"""
    You are 'Justitia', a sharp AI Legal Co-Counsel.
    
    USER QUERY: {history}
    
    COMBINED EVIDENCE:
    {full_evidence}
    
    INSTRUCTIONS:
    1. Answer directly and concisely.
    2. Cite specific laws/sections from the evidence.
    """
    try:
        response = llm.invoke(prompt)
        return {"messages": [response.content]}
    except Exception as e:
        return {"messages": [f"Error generating advice: {str(e)}"]}

# --- 6. WORKFLOW ---
workflow = StateGraph(AgentState)
workflow.add_node("intake", intake)
workflow.add_node("legal_clerk", legal_clerk)
workflow.add_node("evidence_auditor", evidence_auditor)
workflow.add_node("senior_counsel", senior_counsel)

workflow.set_entry_point("intake")

workflow.add_conditional_edges(
    "intake",
    route_logic, 
    ["legal_clerk", "evidence_auditor", "senior_counsel"]
)

workflow.add_edge("legal_clerk", "senior_counsel")
workflow.add_edge("evidence_auditor", "senior_counsel")
workflow.add_edge("senior_counsel", END)

app = workflow.compile()