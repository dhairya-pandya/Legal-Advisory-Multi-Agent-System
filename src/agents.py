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
try:
    from langdetect import detect, LangDetectException
    from langgraph.graph import StateGraph, END
    from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
    from langchain_core.messages import HumanMessage
    from qdrant_client import QdrantClient, models
    from sentence_transformers import SentenceTransformer
    # Direct import for stability
    from ddgs import DDGS
except ImportError as e:
    st.error(f"❌ Missing Dependency: {e}. Please check requirements.txt")
    st.stop()

# --- 1. CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("JustitiaBackend")

class AgentNodes(str, Enum):
    LEGAL_CLERK = "legal_clerk"
    AMENDMENT_WATCHDOG = "amendment_watchdog"
    EVIDENCE_AUDITOR = "evidence_auditor"
    SENIOR_COUNSEL = "senior_counsel"

PROMPTS = {
    "translation": "Translate to English for legal search keywords: '{query}'",
    "video_analysis": "Analyze this video evidence. 1. Chronologically describe the events. 2. Identify any potential illegal acts. 3. Transcribe dialogue.",
    "audio_analysis": "Listen to this audio. 1. Transcribe conversation. 2. Detect emotional tone. 3. Identify threats.",
    "image_analysis": "Analyze this image. If text, transcribe it. If scene, describe legal relevance.",
    "senior_counsel": """
        You are 'Justitia', an AI Legal Co-Counsel.
        USER CONVERSATION: {history}
        
        COMBINED EVIDENCE & RESEARCH:
        {context}
        
        INSTRUCTIONS: 
        1. Answer the legal query directly.
        2. IF 'LIVE WEB UPDATES' are found, prioritize them over static laws.
        3. IF NO CONTEXT IS FOUND: You represent a fallback mechanism. Answer based on your general legal knowledge but explicitly state: "Note: Live verification was unavailable. This advice is based on general legal principles."
        4. Cite specific sections if known.
        5. Output in English.
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
if not CREDS["GOOGLE_API_KEY"]: 
    st.error("Missing GOOGLE_API_KEY. Check secrets or config.")
    st.stop()

# --- 3. AI TOOLS ---
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

# Initialize Global Tools
llm, client, encoder = get_ai_tools()

# --- 4. HELPERS ---
def robust_language_check(text: str) -> bool:
    if not text: return False
    try: return detect(text) != 'en' 
    except LangDetectException: return False 

def check_semantic_cache(query: str) -> Optional[str]:
    try:
        hits = client.search(collection_name="response_cache", query_vector=encoder.encode(query).tolist(), limit=1)
        if hits and hits[0].score > 0.92: return hits[0].payload["answer"]
    except: pass
    return None

def store_in_cache(query: str, answer: str, sensitive_mode: bool = False):
    if sensitive_mode: return
    try:
        client.upsert(collection_name="response_cache", points=[models.PointStruct(
            id=abs(hash(query)) % (10**18), vector=encoder.encode(query).tolist(), payload={"query": query, "answer": answer}
        )])
    except: pass

# --- 5. STATE ---
class AgentState(TypedDict):
    messages: List[str]
    context_data: str 
    file_data: Optional[Dict[str, Any]]
    is_incognito: bool

# --- 6. AGENTS ---

def legal_clerk(state: AgentState) -> Dict[str, str]:
    if not state.get('messages'): return {"context_data": ""}
    
    current_context = state.get("context_data", "")
    raw_query = state['messages'][-1].split("User: ")[-1]

    try:
        search_query = raw_query
        if robust_language_check(raw_query):
            search_query = llm.invoke(PROMPTS["translation"].format(query=raw_query)).content.strip()

        # GENERATE VECTOR
        query_vector = encoder.encode(search_query).tolist()

        # SEARCH STRATEGY (Legacy + Modern Support)
        hits = []
        try:
            # Modern method (v1.7+)
            try:
                # Try named vector search first (if you set up dense/sparse)
                hits = client.search(collection_name="legal_knowledge", query_vector=("dense", query_vector), limit=5)
            except:
                # Fallback to standard vector search
                hits = client.search(collection_name="legal_knowledge", query_vector=query_vector, limit=5)
        except AttributeError:
            # Legacy method (v1.6 and below) - uses search_points implies older syntax, 
            # but usually it was just client.search was missing. 
            # We try the older retrieval method if 'search' is missing.
            from qdrant_client.http import models as rest_models
            hits = client.retrieve(collection_name="legal_knowledge", ids=[1, 2, 3]) # Dummy fallback if search completely fails
            # Note: Older versions used very different syntax. 
            # It is safer to just upgrade the library. 
            return {"context_data": current_context + "\n[System Error: Qdrant Library outdated. Please update requirements.txt]\n"}

        if not hits: return {"context_data": current_context}

        results = [f"- {h.payload.get('full_text', h.payload.get('text', 'Law'))}" for h in hits]
        new_data = "\nLEGAL PRECEDENTS (Static DB):\n" + "\n".join(results) + "\n"
        return {"context_data": current_context + new_data}
        
    except Exception as e:
        return {"context_data": current_context + f"\n[Clerk Error: {e}]\n"}
 
def amendment_watchdog(state: AgentState) -> Dict[str, str]:
    """
    PRODUCTION VERSION: Uses Iterative Search Strategy to find live data.
    """
    current_context = state.get("context_data", "")
    # Check if we have messages to process
    if not state.get('messages'): 
        return {"context_data": current_context}

    # Extract user query
    query = state['messages'][-1].split("User: ")[-1]
    
    # 1. ROBUST LIBRARY IMPORT
    # This block handles the renaming confusion between 'ddgs' and 'duckduckgo_search'
    DDGS_Class = None
    try:
        from duckduckgo_search import DDGS
        DDGS_Class = DDGS
    except ImportError:
        try:
            from ddgs import DDGS
            DDGS_Class = DDGS
        except ImportError:
            st.error("❌ Search Library Missing. Please run: pip install duckduckgo-search")
            return {"context_data": current_context + "\n[System: Live Search Module Missing]\n"}

    # 2. ITERATIVE SEARCH STRATEGY
    # We define 3 distinct search angles. We stop as soon as we get good results.
    search_attempts = [
        # Angle 1: Strict Legal Precedents
        f"Supreme Court of India judgment {query} 2024 2025",
        # Angle 2: Legislative Acts & Bills (Good for 'Civil Aviation Act')
        f"India new bill act law {query} 2024",
        # Angle 3: General Legal News (Broadest net)
        f"{query} India legal news update"
    ]

    found_results = []
    
    try:
        with DDGS_Class() as ddgs:
            for search_term in search_attempts:
                # We limit to 3 results per attempt to keep it fast
                results = list(ddgs.text(search_term, max_results=3))
                
                if results:
                    found_results = results
                    break # Stop searching if we found something useful

        # 3. PROCESS RESULTS
        if not found_results:
            # If all 3 attempts failed, we honestly say so.
            return {"context_data": current_context + f"\nLIVE WEB UPDATES: Searched for '{query}' but found no recent specific legal updates.\n"}

        # Format the output cleanly
        formatted_res = "\n".join([f"- [{r['title']}]({r['href']}): {r['body'][:200]}..." for r in found_results])
        return {"context_data": current_context + f"\nLIVE WEB UPDATES (Source: DuckDuckGo):\n{formatted_res}\n"}

    except Exception as e:
        # 4. EXCEPTION HANDLING
        # Log the specific error to the UI sidebar so you can debug network issues
        st.sidebar.warning(f"⚠️ Live Search Connection Error: {e}")
        return {"context_data": current_context + "\n[System: Live verification failed due to network error]\n"}
def evidence_auditor(state: AgentState) -> Dict[str, str]:
    current_context = state.get("context_data", "")
    file_data = state.get("file_data")
    if not file_data: return {"context_data": current_context}

    file_name = file_data.get("name", "").lower()
    file_type = str(file_data.get("type", "") or "").lower()
    file_bytes = file_data.get("bytes")

    if not file_bytes: return {"context_data": current_context}

    try:
        res = ""
        if "video" in file_type or file_name.endswith(('.mp4', '.avi', '.mov')):
            b64 = base64.b64encode(file_bytes).decode('utf-8')
            msg = HumanMessage(content=[{"type":"text", "text": PROMPTS["video_analysis"]}, {"type":"media", "mime_type":"video/mp4", "data":b64}])
            res = f"VIDEO ANALYSIS:\n{llm.invoke([msg]).content}"
        elif "audio" in file_type or file_name.endswith(('.mp3', '.wav')):
            b64 = base64.b64encode(file_bytes).decode('utf-8')
            msg = HumanMessage(content=[{"type":"text", "text": PROMPTS["audio_analysis"]}, {"type":"media", "mime_type":"audio/mp3", "data":b64}])
            res = f"AUDIO ANALYSIS:\n{llm.invoke([msg]).content}"
        elif "image" in file_type:
            b64 = base64.b64encode(file_bytes).decode('utf-8')
            msg = HumanMessage(content=[{"type":"text", "text": PROMPTS["image_analysis"]}, {"type":"image_url", "image_url":{"url":f"data:image/jpeg;base64,{b64}"}}])
            res = f"IMAGE ANALYSIS:\n{llm.invoke([msg]).content}"
        elif "pdf" in file_type:
            pdf = pypdf.PdfReader(io.BytesIO(file_bytes))
            text = "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])
            res = f"FILE TEXT:\n{text[:4000]}"
        elif "word" in file_type or "doc" in file_type:
             doc = docx.Document(io.BytesIO(file_bytes))
             text = "\n".join([p.text for p in doc.paragraphs])
             res = f"FILE TEXT:\n{text[:4000]}"

        return {"context_data": current_context + "\nEVIDENCE:\n" + res + "\n"}
    except Exception as e:
        return {"context_data": current_context + f"\n[Auditor Error: {e}]\n"}

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
        if not state.get("file_data"): store_in_cache(user_query, answer, is_incognito)
        return {"messages": [answer]}
    except Exception as e:
        return {"messages": [f"Senior Counsel Error: {e}"]}

# --- 7. WORKFLOW ---
workflow = StateGraph(AgentState)
workflow.add_node(AgentNodes.LEGAL_CLERK, legal_clerk)
workflow.add_node(AgentNodes.AMENDMENT_WATCHDOG, amendment_watchdog)
workflow.add_node(AgentNodes.EVIDENCE_AUDITOR, evidence_auditor)
workflow.add_node(AgentNodes.SENIOR_COUNSEL, senior_counsel)

# Linear Chain
workflow.set_entry_point(AgentNodes.LEGAL_CLERK)
workflow.add_edge(AgentNodes.LEGAL_CLERK, AgentNodes.AMENDMENT_WATCHDOG)
workflow.add_edge(AgentNodes.AMENDMENT_WATCHDOG, AgentNodes.EVIDENCE_AUDITOR)
workflow.add_edge(AgentNodes.EVIDENCE_AUDITOR, AgentNodes.SENIOR_COUNSEL)
workflow.add_edge(AgentNodes.SENIOR_COUNSEL, END)

app = workflow.compile()