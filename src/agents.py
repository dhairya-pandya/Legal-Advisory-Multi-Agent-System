import os
import logging
import base64
import io
from enum import Enum
from typing import TypedDict, List, Optional, Dict, Any
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

# --- 1. CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("JustitiaBackend")

class AgentNodes(str, Enum):
    INTAKE = "intake_router"
    LEGAL_CLERK = "legal_clerk"
    EVIDENCE_AUDITOR = "evidence_auditor"
    SENIOR_COUNSEL = "senior_counsel"

PROMPTS = {
    "translation": "Translate to English for legal search keywords: '{query}'",
    "video_analysis": "Analyze this video evidence. 1. Chronologically describe the events. 2. Identify any potential illegal acts (assault, theft, negligence). 3. Transcribe any audible dialogue.",
    "audio_analysis": "Listen to this audio evidence. 1. Transcribe the conversation accurately. 2. Identify the emotional tone (threatening, scared, calm). 3. Identify potential threats.",
    "image_analysis": "Analyze this image. If it is a document, transcribe the text. If it is a scene, describe it relevant to a legal case.",
    "senior_counsel": """
        You are 'Justitia', an AI Legal Co-Counsel.
        USER CONVERSATION: {history}
        EVIDENCE / ANALYSIS: {context}
        INSTRUCTIONS: 
        1. Answer the legal query directly.
        2. Cite specific IPC/BNS sections from the evidence or database.
        3. Output in English (Translation is handled separately).
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

# --- 3. TOOLS ---
@st.cache_resource
def get_ai_tools():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=CREDS["GOOGLE_API_KEY"],
        temperature=0.3,
        safety_settings={HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}
    )
    client = QdrantClient(url=CREDS["QDRANT_URL"], api_key=CREDS["QDRANT_API_KEY"], prefer_grpc=False, timeout=60)
    try: client.get_collection("response_cache")
    except: client.create_collection("response_cache", vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE))
    encoder = SentenceTransformer(CREDS["EMBEDDING_MODEL"])
    return llm, client, encoder

llm, client, encoder = get_ai_tools()

# --- 4. HELPERS ---
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

# --- 5. STATE (Simplified) ---
class AgentState(TypedDict):
    messages: List[str]
    context_data: str  # <--- Back to String (Safe)
    file_data: Optional[Dict[str, Any]]
    current_agent: str
    is_incognito: bool

# --- 6. AGENTS ---

def intake_router(state: AgentState) -> Dict[str, str]:
    if state.get("file_data"):
        return {"current_agent": AgentNodes.EVIDENCE_AUDITOR}
    if not state.get('messages'):
        return {"current_agent": AgentNodes.SENIOR_COUNSEL}
    return {"current_agent": AgentNodes.LEGAL_CLERK}

def legal_clerk(state: AgentState) -> Dict[str, str]:
    if not state.get('messages'): return {"context_data": "No query."}
    
    raw_query = state['messages'][-1].split("User: ")[-1]
    search_query = raw_query
    
    try:
        if robust_language_check(raw_query):
            search_query = llm.invoke(PROMPTS["translation"].format(query=raw_query)).content.strip()
        
        hits = client.query_points("legal_knowledge", query=encoder.encode(search_query).tolist(), limit=5, with_payload=True).points
        if not hits: return {"context_data": f"No specific laws found for: {search_query}"}
        
        results = [f"- {h.payload.get('full_text', 'Law')}" for h in hits]
        return {"context_data": "LEGAL PRECEDENTS (Database):\n" + "\n".join(results)}
    except Exception as e:
        return {"context_data": f"Database Error: {e}"}

def evidence_auditor(state: AgentState) -> Dict[str, str]:
    file_data = state.get("file_data")
    if not file_data: return {"context_data": "No file."}
    
    # SAFETY: Force strings
    file_name = str(file_data.get("name", "")).lower()
    file_type = str(file_data.get("type", "")).lower()
    file_bytes = file_data.get("bytes")

    if not file_bytes: return {"context_data": "Empty file."}

    try:
        # Video
        if "video" in file_type or file_name.endswith(('.mp4', '.avi', '.mov')):
            b64 = base64.b64encode(file_bytes).decode('utf-8')
            msg = HumanMessage(content=[{"type":"text", "text": PROMPTS["video_analysis"]}, {"type":"media", "mime_type":"video/mp4", "data":b64}])
            res = llm.invoke([msg])
            return {"context_data": f"VIDEO ANALYSIS: {res.content}"}
        
        # Audio
        elif "audio" in file_type or file_name.endswith(('.mp3', '.wav')):
            b64 = base64.b64encode(file_bytes).decode('utf-8')
            msg = HumanMessage(content=[{"type":"text", "text": PROMPTS["audio_analysis"]}, {"type":"media", "mime_type":"audio/mp3", "data":b64}])
            res = llm.invoke([msg])
            return {"context_data": f"AUDIO ANALYSIS: {res.content}"}
        
        # Image
        elif "image" in file_type:
            b64 = base64.b64encode(file_bytes).decode('utf-8')
            msg = HumanMessage(content=[{"type":"text", "text": PROMPTS["image_analysis"]}, {"type":"image_url", "image_url":{"url":f"data:image/jpeg;base64,{b64}"}}])
            res = llm.invoke([msg])
            return {"context_data": f"IMAGE ANALYSIS: {res.content}"}
            
        # Documents
        elif "pdf" in file_type:
            pdf = pypdf.PdfReader(io.BytesIO(file_bytes))
            text = "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])
            return {"context_data": f"FILE TEXT: {text[:4000]}"}
            
        elif "word" in file_type or "doc" in file_type:
             doc = docx.Document(io.BytesIO(file_bytes))
             text = "\n".join([p.text for p in doc.paragraphs])
             return {"context_data": f"FILE TEXT: {text[:4000]}"}

        return {"context_data": "File processed but type unclear."}
        
    except Exception as e:
        return {"context_data": f"Analysis Failed: {e}"}

def senior_counsel(state: AgentState) -> Dict[str, List[str]]:
    history = state.get('messages', [])
    context = state.get('context_data', "No evidence.")
    is_incognito = state.get("is_incognito", False)
    user_query = history[-1] if history else ""
    
    if not state.get("file_data") and not is_incognito:
        cached = check_semantic_cache(user_query)
        if cached: return {"messages": [f"**(Cached)** {cached}"]}

    try:
        formatted_prompt = PROMPTS["senior_counsel"].format(history=history, context=context)
        answer = llm.invoke(formatted_prompt).content
        
        if not state.get("file_data"):
            store_in_cache(user_query, answer, is_incognito)
            
        return {"messages": [answer]}
    except Exception as e:
        return {"messages": [f"Error: {e}"]}

# --- 7. WORKFLOW (Linear & Safe) ---
workflow = StateGraph(AgentState)
workflow.add_node(AgentNodes.INTAKE, intake_router)
workflow.add_node(AgentNodes.LEGAL_CLERK, legal_clerk)
workflow.add_node(AgentNodes.EVIDENCE_AUDITOR, evidence_auditor)
workflow.add_node(AgentNodes.SENIOR_COUNSEL, senior_counsel)

workflow.set_entry_point(AgentNodes.INTAKE)

workflow.add_conditional_edges(
    AgentNodes.INTAKE, 
    lambda x: x['current_agent'], 
    {
        AgentNodes.LEGAL_CLERK: AgentNodes.LEGAL_CLERK,
        AgentNodes.EVIDENCE_AUDITOR: AgentNodes.EVIDENCE_AUDITOR,
        AgentNodes.SENIOR_COUNSEL: AgentNodes.SENIOR_COUNSEL
    }
)

workflow.add_edge(AgentNodes.LEGAL_CLERK, AgentNodes.SENIOR_COUNSEL)
workflow.add_edge(AgentNodes.EVIDENCE_AUDITOR, AgentNodes.SENIOR_COUNSEL)
workflow.add_edge(AgentNodes.SENIOR_COUNSEL, END)

app = workflow.compile()