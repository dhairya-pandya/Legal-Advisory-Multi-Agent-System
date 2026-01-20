import os
import json
import logging
import io
import base64
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

# --- 1. CONFIGURATION & LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("JustitiaBackend")

# Define Agent Names as Enums to prevent typos
class AgentNodes(str, Enum):
    INTAKE = "intake_router"
    LEGAL_CLERK = "legal_clerk"
    EVIDENCE_AUDITOR = "evidence_auditor"
    SENIOR_COUNSEL = "senior_counsel"

# Centralized Prompts for easy tuning
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

# --- 2. CREDENTIALS MANAGEMENT ---
def get_credentials() -> Dict[str, str]:
    """Retrieves API keys from Config, Secrets, or Environment."""
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

if not CREDS["GOOGLE_API_KEY"]:
    st.error("🚨 Critical Error: GOOGLE_API_KEY is missing. Please check config.py or Streamlit Secrets.")
    st.stop()

# --- 3. AI TOOLS INITIALIZATION (SINGLETON) ---
@st.cache_resource
def get_ai_tools():
    """Initializes and caches heavy AI models."""
    # 1. LLM: Gemini 2.5 Flash (Native Multimodal)
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
    
    # 2. Database: Qdrant
    client = QdrantClient(url=CREDS["QDRANT_URL"], api_key=CREDS["QDRANT_API_KEY"], prefer_grpc=False, timeout=60)
    
    # Ensure 'response_cache' collection exists for optimization
    try:
        client.get_collection("response_cache")
    except Exception:
        logger.info("Creating 'response_cache' collection...")
        client.create_collection(
            collection_name="response_cache",
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
        )

    # 3. Embeddings
    encoder = SentenceTransformer(CREDS["EMBEDDING_MODEL"])
    
    return llm, client, encoder

llm, client, encoder = get_ai_tools()

# --- 4. OPTIMIZATION HELPERS ---

def robust_language_check(text: str) -> bool:
    """
    Check if text is Non-English using local CPU (saves API calls).
    Returns True if translation is needed.
    """
    try:
        lang = detect(text)
        return lang != 'en' 
    except LangDetectException:
        return False 

def check_semantic_cache(query: str) -> Optional[str]:
    """Checks Qdrant for a previously answered similar question."""
    try:
        vector = encoder.encode(query).tolist()
        hits = client.search(
            collection_name="response_cache",
            query_vector=vector,
            limit=1
        )
        # Threshold 0.92 ensures high semantic similarity
        if hits and hits[0].score > 0.92:
            logger.info("CACHE HIT! Returning stored answer.")
            return hits[0].payload["answer"]
    except Exception as e:
        logger.warning(f"Cache Check Failed: {e}")
    return None

def store_in_cache(query: str, answer: str):
    """Stores the Query-Answer pair in Qdrant for future use."""
    try:
        vector = encoder.encode(query).tolist()
        # Use hash of query as ID to prevent duplicates
        point_id = abs(hash(query)) % (10**18) 
        client.upsert(
            collection_name="response_cache",
            points=[
                models.PointStruct(
                    id=point_id, 
                    vector=vector,
                    payload={"query": query, "answer": answer}
                )
            ]
        )
    except Exception as e:
        logger.warning(f"Cache Store Failed: {e}")

# --- 5. STATE MANAGEMENT ---
class AgentState(TypedDict):
    messages: List[str]
    context_data: str 
    file_data: Optional[Dict[str, Any]]
    current_agent: str

# --- 6. AGENT FUNCTIONS ---

def intake_router(state: AgentState) -> Dict[str, str]:
    """Routes execution based on input type."""
    if state.get("file_data"):
        return {"current_agent": AgentNodes.EVIDENCE_AUDITOR}
    if not state.get('messages'):
        return {"current_agent": AgentNodes.SENIOR_COUNSEL}
    return {"current_agent": AgentNodes.LEGAL_CLERK}

def legal_clerk(state: AgentState) -> Dict[str, str]:
    """Retrieves legal context from Qdrant."""
    if not state.get('messages'): return {"context_data": "No query provided."}
    
    raw_query = state['messages'][-1].split("User: ")[-1]
    search_query = raw_query
    
    try:
        # Optimization: Only translate if language is NOT English
        if robust_language_check(raw_query):
            trans_prompt = PROMPTS["translation"].format(query=raw_query)
            trans_res = llm.invoke(trans_prompt)
            search_query = trans_res.content.strip()
        
        # Vector Search
        hits = client.query_points(
            collection_name="legal_knowledge",
            query=encoder.encode(search_query).tolist(),
            using="dense",
            limit=5, 
            with_payload=True
        ).points
        
        if not hits: return {"context_data": f"No specific laws found for: {search_query}"}
        
        results = [f"- {h.payload.get('full_text', h.payload.get('text', 'Law'))}" for h in hits]
        return {"context_data": "LEGAL PRECEDENTS (Database):\n" + "\n".join(results)}

    except Exception as e:
        logger.error(f"Legal Clerk Error: {e}")
        return {"context_data": f"Database Error: {e}"}

def evidence_auditor(state: AgentState) -> Dict[str, str]:
    """Multimodal Analysis: Handles Video, Audio, Images, and Docs."""
    file_data = state.get("file_data")
    if not file_data: return {"context_data": "No file uploaded."}
    
    file_name = file_data["name"].lower()
    file_type = file_data["type"]
    file_bytes = file_data["bytes"]

    try:
        # --- 1. VIDEO HANDLING ---
        if "video" in file_type or file_name.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            b64_video = base64.b64encode(file_bytes).decode('utf-8')
            msg = HumanMessage(content=[
                {"type": "text", "text": PROMPTS["video_analysis"]},
                {"type": "media", "mime_type": "video/mp4", "data": b64_video}
            ])
            res = llm.invoke([msg])
            return {"context_data": f"VIDEO ANALYSIS ({file_name}):\n{res.content}"}

        # --- 2. AUDIO HANDLING ---
        elif "audio" in file_type or file_name.endswith(('.mp3', '.wav', '.m4a')):
            b64_audio = base64.b64encode(file_bytes).decode('utf-8')
            msg = HumanMessage(content=[
                {"type": "text", "text": PROMPTS["audio_analysis"]},
                {"type": "media", "mime_type": "audio/mp3", "data": b64_audio}
            ])
            res = llm.invoke([msg])
            return {"context_data": f"AUDIO ANALYSIS ({file_name}):\n{res.content}"}

        # --- 3. IMAGE HANDLING ---
        elif "image" in file_type or file_name.endswith(('.png', '.jpg', '.jpeg')):
            b64_img = base64.b64encode(file_bytes).decode('utf-8')
            msg = HumanMessage(content=[
                {"type":"text", "text": PROMPTS["image_analysis"]}, 
                {"type":"image_url", "image_url":{"url":f"data:image/jpeg;base64,{b64_img}"}}
            ])
            res = llm.invoke([msg])
            return {"context_data": f"IMAGE ANALYSIS ({file_name}):\n{res.content}"}
        
        # --- 4. DOCUMENT HANDLING ---
        elif "pdf" in file_type:
            pdf = pypdf.PdfReader(io.BytesIO(file_bytes))
            text = "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])
            return {"context_data": f"FILE TEXT ({file_name}):\n{text[:4000]}"}

        elif "word" in file_type or "document" in file_type:
             doc = docx.Document(io.BytesIO(file_bytes))
             text = "\n".join([p.text for p in doc.paragraphs])
             return {"context_data": f"FILE TEXT ({file_name}):\n{text[:4000]}"}

        else:
            return {"context_data": f"Error: Unsupported file type ({file_type})."}
        
    except Exception as e:
        logger.error(f"Evidence Auditor Error: {e}")
        return {"context_data": f"Analysis Failed: {e}"}

def senior_counsel(state: AgentState) -> Dict[str, List[str]]:
    """Synthesizes Final Verdict with Caching logic."""
    history = state.get('messages', [])
    context = state.get('context_data', "No evidence.")
    
    user_query = history[-1] if history else ""
    
    # --- CACHE READ ---
    # Only check cache if no file (evidence makes queries unique)
    if not state.get("file_data"):
        cached_ans = check_semantic_cache(user_query)
        if cached_ans:
            return {"messages": [f"**(Cached)** {cached_ans}"]}

    # --- GENERATE ---
    formatted_prompt = PROMPTS["senior_counsel"].format(history=history, context=context)
    
    try:
        res = llm.invoke(formatted_prompt)
        answer = res.content
        
        # --- CACHE WRITE ---
        if not state.get("file_data"):
            store_in_cache(user_query, answer)
            
        return {"messages": [answer]}
    except Exception as e:
        logger.error(f"Senior Counsel Error: {e}")
        return {"messages": ["Error generating advice."]}

# --- 7. WORKFLOW CONSTRUCTION ---
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