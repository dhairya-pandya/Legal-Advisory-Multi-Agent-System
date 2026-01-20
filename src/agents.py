import os
import sys
import io
import base64
import streamlit as st
from typing import TypedDict, List, Optional
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

if not CREDS["GOOGLE_API_KEY"]:
    st.error("🚨 Configuration Error: API Keys not found.")
    st.stop()

# --- 2. AI TOOLS INITIALIZATION ---
@st.cache_resource
def get_ai_tools():
    # Using Gemini 2.5 Flash for Native Multimodal (Video/Audio) support
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

# --- 3. AGENT STATE DEFINITION ---
class AgentState(TypedDict):
    messages: List[str]
    context_data: str 
    file_data: Optional[dict]
    current_agent: str

# --- 4. AGENTS ---

def intake_router(state: AgentState):
    """Decides which agent to run based on input type."""
    if state.get("file_data"):
        return {"current_agent": "evidence_auditor"}
    if not state['messages']:
        return {"current_agent": "senior_counsel"}
    # Default to Legal Search for text queries
    return {"current_agent": "legal_clerk"}

def legal_clerk(state: AgentState):
    """The Researcher: Searches Qdrant for Laws."""
    if not state.get('messages'): return {"context_data": "No query."}
    raw_query = state['messages'][-1].split("User: ")[-1]
    
    try:
        # Internal thought: Translate to English for better search precision
        trans_prompt = f"Translate to English for legal search keywords: '{raw_query}'"
        search_query = llm.invoke(trans_prompt).content.strip()
        
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
        return {"context_data": f"Database Error: {e}"}

def evidence_auditor(state: AgentState):
    """The Forensics Expert: Analyzes Video, Audio, Images, Docs."""
    file_data = state.get("file_data")
    if not file_data: return {"context_data": "No file."}
    
    file_name = file_data["name"].lower()
    file_type = file_data["type"]
    file_bytes = file_data["bytes"]

    try:
        # A. VIDEO ANALYSIS (Native Gemini 2.5)
        if "video" in file_type or file_name.endswith(('.mp4', '.avi', '.mov')):
            b64_video = base64.b64encode(file_bytes).decode('utf-8')
            msg = HumanMessage(content=[
                {"type": "text", "text": "Analyze this video evidence. 1. Chronologically describe the events. 2. Identify any illegal acts (assault, theft, negligence). 3. Transcribe audible dialogue."},
                {"type": "media", "mime_type": "video/mp4", "data": b64_video}
            ])
            res = llm.invoke([msg])
            return {"context_data": f"VIDEO EVIDENCE ANALYSIS ({file_name}):\n{res.content}"}

        # B. AUDIO ANALYSIS (Native Gemini 2.5)
        elif "audio" in file_type or file_name.endswith(('.mp3', '.wav')):
            b64_audio = base64.b64encode(file_bytes).decode('utf-8')
            msg = HumanMessage(content=[
                {"type": "text", "text": "Listen to this audio. 1. Transcribe the conversation. 2. Detect the tone/emotion. 3. Identify potential threats or admissions."},
                {"type": "media", "mime_type": "audio/mp3", "data": b64_audio}
            ])
            res = llm.invoke([msg])
            return {"context_data": f"AUDIO EVIDENCE ANALYSIS ({file_name}):\n{res.content}"}

        # C. IMAGE ANALYSIS
        elif "image" in file_type or file_name.endswith(('.png', '.jpg', '.jpeg')):
            b64_img = base64.b64encode(file_bytes).decode('utf-8')
            msg = HumanMessage(content=[
                {"type":"text", "text":"Analyze this image. If it's a document, transcribe it. If it's a photo, describe the scene relevant to a legal case."}, 
                {"type":"image_url", "image_url":{"url":f"data:image/jpeg;base64,{b64_img}"}}
            ])
            res = llm.invoke([msg])
            return {"context_data": f"IMAGE ANALYSIS:\n{res.content}"}
        
        # D. PDF/DOCS HANDLING
        elif "pdf" in file_type:
            pdf = pypdf.PdfReader(io.BytesIO(file_bytes))
            text = "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])
            return {"context_data": f"FILE TEXT:\n{text[:4000]}"}

        return {"context_data": "File content extracted."}
        
    except Exception as e:
        return {"context_data": f"File Analysis Error: {e}"}

def senior_counsel(state: AgentState):
    """The Judge: Synthesizes final advice in English."""
    history = state.get('messages', [])
    context = state.get('context_data', "No evidence.")
    
    prompt = f"""
    You are 'Justitia', an AI Legal Co-Counsel.
    USER CONVERSATION: {history}
    EVIDENCE / ANALYSIS: {context}
    INSTRUCTIONS: 
    1. Answer the legal query directly.
    2. Cite specific IPC/BNS sections from the evidence or database.
    3. Output in English (Translation is handled separately).
    """
    try:
        res = llm.invoke(prompt)
        return {"messages": [res.content]}
    except Exception as e:
        return {"messages": [f"Error: {e}"]}

# --- 5. WORKFLOW GRAPH ---
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