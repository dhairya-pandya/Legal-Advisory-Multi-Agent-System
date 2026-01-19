import os
import sys
import io
import base64
import operator
import streamlit as st
from typing import TypedDict, List, Optional, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain_core.messages import HumanMessage
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from flashrank import Ranker, RerankRequest
from tenacity import retry, stop_after_attempt, wait_random_exponential
import pypdf
import docx

# --- 1. CONFIG SETUP ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from config import GOOGLE_API_KEY, QDRANT_URL, QDRANT_API_KEY, EMBEDDING_MODEL
except ImportError:
    from src.config import GOOGLE_API_KEY, QDRANT_URL, QDRANT_API_KEY, EMBEDDING_MODEL

# --- 2. CACHED TOOLS ---
@st.cache_resource
def get_ai_tools():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }
    )
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=False, timeout=60)
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank")
    return llm, client, encoder, ranker

llm, client, encoder, ranker = get_ai_tools()

# --- 3. AGENT STATE (PARALLEL ENABLED) ---
class AgentState(TypedDict):
    messages: List[str]
    # PARALLEL FIX: Use Annotated + operator.add to merge results from multiple agents
    context_data: Annotated[List[str], operator.add] 
    file_data: Optional[dict]

# --- 4. AGENTS ---

def intake_router(state: AgentState) -> List[str]:
    """
    Fan-Out Router: Returns a LIST of agents to run in parallel.
    """
    agents_to_run = []

    # 1. If User provided text, we need the Legal Clerk
    # (Checking if the last message is from user and has content)
    if state['messages']:
        agents_to_run.append("legal_clerk")

    # 2. If User uploaded a file, we need the Evidence Auditor
    if state.get("file_data"):
        agents_to_run.append("evidence_auditor")

    # Fallback: If nothing to do, go straight to Senior Counsel (should rarely happen)
    if not agents_to_run:
        return ["senior_counsel"]
        
    return agents_to_run

def legal_clerk(state: AgentState):
    """Agent B: Research Agent (Returns List for Merge)."""
    if not state['messages']:
        return {"context_data": []}
        
    query = state['messages'][-1].split("User: ")[-1] # Extract actual query if needed
    
    try:
        # 1. Search
        hits = client.query_points(
            collection_name="legal_knowledge",
            query=encoder.encode(query).tolist(),
            using="dense",
            limit=15, 
            with_payload=True
        ).points
        
        if not hits:
            return {"context_data": ["Legal Clerk: No specific laws found."]}

        # 2. Rerank
        passages = [{"id": h.id, "text": h.payload.get('full_text', '')} for h in hits]
        results = ranker.rerank(RerankRequest(query=query, passages=passages))
        
        final_results = []
        for res in results[:5]:
            if res['score'] > 0.4:
                final_results.append(f"- [Rel: {res['score']:.2f}] {res['text']}")
        
        if not final_results:
             return {"context_data": ["Legal Clerk: Found laws but low relevance."]}
            
        # RETURN AS LIST (Crucial for operator.add)
        combined_text = "LEGAL PRECEDENTS FOUND:\n" + "\n".join(final_results)
        return {"context_data": [combined_text]}

    except Exception as e:
        return {"context_data": [f"Legal Clerk Error: {e}"]}

# Retry Logic Wrapper
@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
def robust_llm_invoke(messages):
    return llm.invoke(messages)

def evidence_auditor(state: AgentState):
    """Agent C: Vision Specialist (Returns List for Merge)."""
    file_data = state.get("file_data")
    if not file_data:
        return {"context_data": []}

    file_name = file_data["name"]
    file_type = file_data["type"]
    file_bytes = file_data["bytes"]
    
    try:
        if "image" in file_type or file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            b64_image = base64.b64encode(file_bytes).decode('utf-8')
            msg = HumanMessage(content=[
                {"type": "text", "text": "Forensic Analysis: 1. Transcribe text. 2. Flag issues."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
            ])
            response = robust_llm_invoke([msg])
            return {"context_data": [f"EVIDENCE ANALYSIS ({file_name}):\n{response.content}"]}

        elif "pdf" in file_type:
            pdf = pypdf.PdfReader(io.BytesIO(file_bytes))
            text = "\n".join([p.extract_text() for p in pdf.pages])
            return {"context_data": [f"FILE TEXT ({file_name}):\n{text[:5000]}"]}
            
        elif "word" in file_type:
            doc = docx.Document(io.BytesIO(file_bytes))
            text = "\n".join([p.text for p in doc.paragraphs])
            return {"context_data": [f"FILE TEXT ({file_name}):\n{text[:5000]}"]}
            
        return {"context_data": ["Evidence Auditor: Unsupported file type."]}

    except Exception as e:
        return {"context_data": [f"Evidence Auditor Error: {e}"]}

def senior_counsel(state: AgentState):
    """Agent D: Fan-In Synthesizer."""
    history = state['messages']
    # JOIN the list of contexts into one big string for the LLM
    context_list = state.get('context_data', [])
    full_evidence = "\n\n".join(context_list) if context_list else "No evidence provided."
    
    prompt = f"""
    You are 'Justitia', a sharp AI Legal Co-Counsel.
    
    USER QUERY: {history}
    
    COMBINED EVIDENCE (From Parallel Agents):
    {full_evidence}
    
    INSTRUCTIONS:
    1. Answer directly.
    2. Synthesize findings from BOTH the Legal Precedents and the Evidence Analysis (if present).
    3. If the user's document contradicts the law (or itself), point it out.
    """
    try:
        response = robust_llm_invoke(prompt)
        return {"messages": [response.content]}
    except Exception as e:
        return {"messages": [f"Error generating advice: {e}"]}

# --- 5. PARALLEL WORKFLOW ---
workflow = StateGraph(AgentState)
workflow.add_node("intake", intake_router)
workflow.add_node("legal_clerk", legal_clerk)
workflow.add_node("evidence_auditor", evidence_auditor)
workflow.add_node("senior_counsel", senior_counsel)

workflow.set_entry_point("intake")

# THE PARALLEL EDGE
# The router returns a list ["legal_clerk", "evidence_auditor"]
# LangGraph runs them both, then moves to "senior_counsel" when BOTH are done.
workflow.add_conditional_edges(
    "intake",
    intake_router,
    ["legal_clerk", "evidence_auditor", "senior_counsel"]
)

workflow.add_edge("legal_clerk", "senior_counsel")
workflow.add_edge("evidence_auditor", "senior_counsel")
workflow.add_edge("senior_counsel", END)

app = workflow.compile()