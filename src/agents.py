import os
import logging
import base64
import io
import operator  # <--- CRITICAL FOR PARALLEL MERGING
from enum import Enum
from typing import TypedDict, List, Optional, Dict, Any, Annotated
import streamlit as st
import pypdf
import docx

# --- THIRD PARTY LIBRARIES ---
from langdetect import detect, LangDetectException
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain_core.messages import HumanMessage
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from langchain_community.tools import DuckDuckGoSearchRun # <--- NEW WATCHDOG TOOL

# --- 1. CONFIGURATION & LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("JustitiaBackend")

class AgentNodes(str, Enum):
    INTAKE = "intake_router"
    LEGAL_CLERK = "legal_clerk"
    AMENDMENT_WATCHDOG = "amendment_watchdog" # <--- NEW AGENT
    EVIDENCE_AUDITOR = "evidence_auditor"
    SENIOR_COUNSEL = "senior_counsel"

# Centralized Prompts
PROMPTS = {
    "translation": "Translate to English for legal search keywords: '{query}'",
    "video_analysis": "Analyze this video evidence. 1. Chronologically describe the events. 2. Identify any potential illegal acts (assault, theft, negligence). 3. Transcribe any audible dialogue.",
    "audio_analysis": "Listen to this audio evidence. 1. Transcribe the conversation accurately. 2. Identify the emotional tone (threatening, scared, calm). 3. Identify potential threats.",
    "image_analysis": "Analyze this image. If it is a document, transcribe the text. If it is a scene, describe it relevant to a legal case.",
    "senior_counsel": """
        You are 'Justitia', an AI Legal Co-Counsel.
        USER CONVERSATION: {history}
        
        COMBINED EVIDENCE & RESEARCH:
        {context}
        
        INSTRUCTIONS: 
        1. Answer the legal query directly.
        2. IF 'LIVE WEB UPDATES' contradicts 'LEGAL PRECEDENTS', prioritize the Live Web Updates (Recent Amendments/BNS).
        3. Cite specific sections.
        4. Output in English.
    """
}

# --- 2. CREDENTIALS ---
def get_credentials() -> Dict[str, str]:
    creds = {}
    try:
        import config
        creds["GOOGLE_API_KEY"] = getattr(config, "GOOGLE_API_KEY", None)
        creds["QDRANT_URL"] = getattr(config, "QDRANT_URL", None)
        creds["QDRANT_API_KEY"] = getattr(config, "QDRANT_API_KEY", None)
        creds["EMBEDDING_MODEL"] = getattr(config, "EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    except ImportError: pass
    
    keys = ["GOOGLE_API_KEY", "QDRANT_URL", "QDRANT_API_KEY", "EMBEDDING_MODEL"]
    for k in keys:
        if not creds.get(k): creds[k] = st.secrets.get(k, os.getenv(k))
    return creds

CREDS = get_credentials()
if not CREDS["GOOGLE_API_KEY"]: st.stop()

# --- 3. AI TOOLS INITIALIZATION ---
@st.cache_resource
def get_ai_tools():
    # 1. LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=CREDS["GOOGLE_API_KEY"],
        temperature=0.3,
        safety_settings={HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}
    )
    
    # 2. Database
    client = QdrantClient(url=CREDS["QDRANT_URL"], api_key=CREDS["QDRANT_API_KEY"], prefer_grpc=False, timeout=60)
    try: client.get_collection("response_cache")
    except: client.create_collection("response_cache", vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE))

    # 3. Embeddings & Search
    encoder = SentenceTransformer(CREDS["EMBEDDING_MODEL"])
    web_search = DuckDuckGoSearchRun() # <--- Live Internet Search
    
    return llm, client, encoder, web_search

llm, client, encoder, web_search = get_ai_tools()

# --- 4. OPTIMIZATION HELPERS ---
def robust_language_check(text: str) -> bool:
    try: return detect(text) != 'en' 
    except LangDetectException: return False 

def check_semantic_cache(query: str) -> Optional[str]:
    try:
        hits = client.search("response_cache", query_vector=encoder.encode(query).tolist(), limit=1)
        if hits and hits[0].score > 0.92: return hits[0].payload["answer"]
    except: pass
    return None

def store_in_cache(query: str, answer: str, sensitive_mode: bool = False):
    if sensitive_mode: return
    try:
        client.upsert("response_cache", points=[models.PointStruct(
            id=abs(hash(query)) % (10**18), 
            vector=encoder.encode(query).tolist(), 
            payload={"query": query, "answer": answer}
        )])
    except: pass

# --- 5. STATE MANAGEMENT (PARALLEL FIX) ---
class AgentState(TypedDict):
    messages: List[str]
    # CRITICAL: Annotated list with reducer allows parallel agents to write simultaneously without overwriting
    context_data: Annotated[List[str], operator.add] 
    file_data: Optional[Dict[str, Any]]
    is_incognito: bool

# --- 6. AGENT FUNCTIONS ---

def parallel_router(state: AgentState) -> List[str]:
    """
    Returns a LIST of agents to run in parallel.
    Fan-Out Pattern.
    """
    agents = []
    
    # If there is a text query, we need BOTH Static Law (Clerk) and Live Updates (Watchdog)
    if state.get('messages'):
        agents.append(AgentNodes.LEGAL_CLERK)
        agents.append(AgentNodes.AMENDMENT_WATCHDOG)
        
    # If there is a file, we need the Auditor
    if state.get("file_data"):
        agents.append(AgentNodes.EVIDENCE_AUDITOR)
        
    # Fallback to Senior Counsel if empty (rare)
    if not agents:
        return [AgentNodes.SENIOR_COUNSEL]
        
    return agents

def legal_clerk(state: AgentState) -> Dict[str, List[str]]:
    """Agent A: Static Database Search"""
    if not state.get('messages'): return {"context_data": []}
    
    raw_query = state['messages'][-1].split("User: ")[-1]
    search_query = raw_query
    
    try:
        if robust_language_check(raw_query):
            search_query = llm.invoke(PROMPTS["translation"].format(query=raw_query)).content.strip()
        
        hits = client.query_points("legal_knowledge", query=encoder.encode(search_query).tolist(), limit=5, with_payload=True).points
        if not hits: return {"context_data": [f"Legal Clerk: No specific laws found for {search_query}."]}
        
        results = [f"- {h.payload.get('full_text', 'Law')}" for h in hits]
        # RETURN AS LIST
        return {"context_data": ["LEGAL PRECEDENTS (Static DB):\n" + "\n".join(results)]}
    except Exception as e:
        return {"context_data": [f"Legal Clerk Error: {e}"]}

def amendment_watchdog(state: AgentState) -> Dict[str, List[str]]:
    """Agent E: Live Internet Search (The New Agent)"""
    if not state.get('messages'): return {"context_data": []}
    
    query = state['messages'][-1].split("User: ")[-1]
    try:
        # We construct a query specifically for updates/amendments
        search_prompt = f"latest legal amendments supreme court judgments {query} India 2024 2025"
        live_results = web_search.invoke(search_prompt)
        # RETURN AS LIST
        return {"context_data": [f"LIVE WEB UPDATES (Watchdog):\n{live_results}"]}
    except Exception as e:
        return {"context_data": [f"Watchdog Notice: Live search unavailable ({e})"]}

def evidence_auditor(state: AgentState) -> Dict[str, List[str]]:
    """Agent C: Multimodal Analysis"""
    file_data = state.get("file_data")
    if not file_data: return {"context_data": []}
    
    file_name = file_data["name"].lower()
    file_type = file_data["type"]
    file_bytes = file_data["bytes"]

    try:
        analysis_result = ""
        # Video
        if "video" in file_type or file_name.endswith(('.mp4', '.avi', '.mov')):
            b64 = base64.b64encode(file_bytes).decode('utf-8')
            msg = HumanMessage(content=[{"type":"text", "text": PROMPTS["video_analysis"]}, {"type":"media", "mime_type":"video/mp4", "data":b64}])
            analysis_result = f"VIDEO ANALYSIS ({file_name}):\n{llm.invoke([msg]).content}"
            
        # Audio
        elif "audio" in file_type or file_name.endswith(('.mp3', '.wav')):
            b64 = base64.b64encode(file_bytes).decode('utf-8')
            msg = HumanMessage(content=[{"type":"text", "text": PROMPTS["audio_analysis"]}, {"type":"media", "mime_type":"audio/mp3", "data":b64}])
            analysis_result = f"AUDIO ANALYSIS ({file_name}):\n{llm.invoke([msg]).content}"
            
        # Image
        elif "image" in file_type:
            b64 = base64.b64encode(file_bytes).decode('utf-8')
            msg = HumanMessage(content=[{"type":"text", "text": PROMPTS["image_analysis"]}, {"type":"image_url", "image_url":{"url":f"data:image/jpeg;base64,{b64}"}}])
            analysis_result = f"IMAGE ANALYSIS ({file_name}):\n{llm.invoke([msg]).content}"
            
        # PDF
        elif "pdf" in file_type:
            pdf = pypdf.PdfReader(io.BytesIO(file_bytes))
            text = "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])
            analysis_result = f"FILE TEXT ({file_name}):\n{text[:4000]}"
            
        return {"context_data": [analysis_result] if analysis_result else []}
        
    except Exception as e:
        return {"context_data": [f"Evidence Auditor Error: {e}"]}

def senior_counsel(state: AgentState) -> Dict[str, List[str]]:
    """Agent D: The Judge (Fan-In Aggregator)"""
    history = state.get('messages', [])
    
    # AGGREGATION: We join the list of contexts from all parallel agents
    context_list = state.get('context_data', [])
    combined_context = "\n\n".join(context_list) if context_list else "No evidence found."
    
    is_incognito = state.get("is_incognito", False)
    user_query = history[-1] if history else ""
    
    # Cache Read
    if not state.get("file_data") and not is_incognito:
        cached = check_semantic_cache(user_query)
        if cached: return {"messages": [f"**(Cached)** {cached}"]}

    # Synthesis
    try:
        formatted_prompt = PROMPTS["senior_counsel"].format(history=history, context=combined_context)
        answer = llm.invoke(formatted_prompt).content
        
        # Cache Write
        if not state.get("file_data"):
            store_in_cache(user_query, answer, is_incognito)
            
        return {"messages": [answer]}
    except Exception as e:
        return {"messages": [f"Senior Counsel Error: {e}"]}

# --- 7. WORKFLOW (PARALLEL CONFIG) ---
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node(AgentNodes.INTAKE, lambda s: {}) # Dummy start node
workflow.add_node(AgentNodes.LEGAL_CLERK, legal_clerk)
workflow.add_node(AgentNodes.AMENDMENT_WATCHDOG, amendment_watchdog)
workflow.add_node(AgentNodes.EVIDENCE_AUDITOR, evidence_auditor)
workflow.add_node(AgentNodes.SENIOR_COUNSEL, senior_counsel)

# Set Entry
workflow.set_entry_point(AgentNodes.INTAKE)

# Fan-Out Logic
workflow.add_conditional_edges(
    AgentNodes.INTAKE,
    parallel_router,
    # This map allows the router to return a list of these keys
    [AgentNodes.LEGAL_CLERK, AgentNodes.AMENDMENT_WATCHDOG, AgentNodes.EVIDENCE_AUDITOR, AgentNodes.SENIOR_COUNSEL]
)

# Fan-In Logic (All parallel nodes go to Senior Counsel)
workflow.add_edge(AgentNodes.LEGAL_CLERK, AgentNodes.SENIOR_COUNSEL)
workflow.add_edge(AgentNodes.AMENDMENT_WATCHDOG, AgentNodes.SENIOR_COUNSEL)
workflow.add_edge(AgentNodes.EVIDENCE_AUDITOR, AgentNodes.SENIOR_COUNSEL)
workflow.add_edge(AgentNodes.SENIOR_COUNSEL, END)

app = workflow.compile()