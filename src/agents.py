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
from sentence_transformers import SentenceTransformer, CrossEncoder
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
    # 1. LLM: Gemini 1.5 Flash (Vision Enabled + Safety Disabled)
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
    
    # 2. Database: Qdrant
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=False, timeout=60)
    
    # 3. Search Models
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2') 
    
    return llm, client, encoder, reranker

llm, client, encoder, reranker = get_ai_tools()

# --- 3. AGENT STATE ---
class AgentState(TypedDict):
    messages: List[str]
    current_agent: str
    context_data: str
    file_data: Optional[dict] # Holds the uploaded file bytes

# --- 4. AGENTS ---

def intake_router(state: AgentState):
    """Routes based on File Upload OR Keywords."""
    # PRIORITY: If a file is uploaded, go straight to the Auditor
    if state.get("file_data"):
        return {"current_agent": "evidence_auditor"}

    if not state['messages']:
        return {"current_agent": "senior_counsel"}
    
    last_msg = state['messages'][-1].lower()
    
    if any(x in last_msg for x in ["law", "section", "act", "punishment", "ipc", "rights", "crime", "court", "arrest", "police"]):
        return {"current_agent": "legal_clerk"}
    else:
        return {"current_agent": "senior_counsel"}

def legal_clerk(state: AgentState):
    """Agent B: Research Agent (Search + Rerank)."""
    query = state['messages'][-1]
    
    try:
        # Broad Search (Dense)
        hits = client.query_points(
            collection_name="legal_knowledge",
            query=encoder.encode(query).tolist(),
            using="dense",
            limit=15, 
            with_payload=True
        ).points
        
        if not hits:
            return {"context_data": "No matches found.", "messages": ["No specific laws found in database."]}

        # Rerank Results
        docs = [hit.payload.get('full_text', hit.payload.get('text', '')) for hit in hits]
        scores = reranker.predict([[query, doc] for doc in docs])
        ranked_hits = sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)
        
        # Filter Top Results
        final_results = []
        for score, hit in ranked_hits[:5]:
            if score > 0.01:
                text = hit.payload.get('full_text', hit.payload.get('text', ''))
                final_results.append(f"- [Rel: {score:.2f}] {text}")
        
        if not final_results:
             return {"context_data": "Low relevance.", "messages": ["I searched but found no relevant laws."]}
            
        return {"context_data": "\n".join(final_results), "messages": [f"Found {len(final_results)} relevant legal precedents."]}

    except Exception as e:
        return {"context_data": f"Error: {e}", "messages": ["Database access error."]}

def evidence_auditor(state: AgentState):
    """
    Agent C: The Vision & Document Specialist.
    Handles PNG/JPG (Gemini Vision), PDF (PyPDF), Docx (Python-Docx).
    """
    file_data = state.get("file_data")
    if not file_data:
        return {"context_data": "No file.", "messages": ["No file was uploaded."]}

    file_name = file_data["name"]
    file_type = file_data["type"]
    file_bytes = file_data["bytes"]
    
    extracted_content = ""

    try:
        # --- HANDLER 1: IMAGES (Vision Analysis) ---
        if "image" in file_type or file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Encode for Gemini
            b64_image = base64.b64encode(file_bytes).decode('utf-8')
            
            message = HumanMessage(
                content=[
                    {"type": "text", "text": "You are a Forensic Document Examiner. Analyze this image. 1. Transcribe the text. 2. Describe signatures/stamps. 3. Flag any issues."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                ]
            )
            response = llm.invoke([message])
            extracted_content = f"[IMAGE ANALYSIS]:\n{response.content}"

        # --- HANDLER 2: PDF (Text Extraction) ---
        elif "pdf" in file_type or file_name.lower().endswith('.pdf'):
            pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            if len(text.strip()) < 50:
                 extracted_content = f"[PDF SCAN DETECTED]: The file '{file_name}' seems to be a scan. For best results, convert to JPG and upload again."
            else:
                 extracted_content = f"[PDF TEXT]:\n{text}"

        # --- HANDLER 3: DOCX (Text Extraction) ---
        elif "word" in file_type or file_name.lower().endswith('.docx'):
            doc = docx.Document(io.BytesIO(file_bytes))
            text = "\n".join([para.text for para in doc.paragraphs])
            extracted_content = f"[DOCX TEXT]:\n{text}"
            
        else:
            return {"context_data": "Unsupported file.", "messages": ["File type not supported."]}

        return {
            "context_data": extracted_content[:4000], # Keep context manageable
            "messages": [f"I have successfully analyzed the file '{file_name}'."]
        }

    except Exception as e:
        return {"context_data": f"File Error: {e}", "messages": ["Failed to process the file."]}

def senior_counsel(state: AgentState):
    """Agent D: Final Advice Generator."""
    history = state['messages']
    context = state.get('context_data', 'No context.')
    
    prompt = f"""
    You are 'Justitia', an AI Legal Co-Counsel for India.
    
    USER QUERY: {history}
    EVIDENCE / FILE ANALYSIS: {context}
    
    INSTRUCTIONS:
    1. If a file was analyzed, summarize the findings first.
    2. Answer the user's query using the Legal Evidence/File Analysis.
    3. Cite specific Acts/Sections if available.
    4. Warn if the file quality (scans/images) might affect accuracy.
    """
    try:
        response = llm.invoke(prompt)
        return {"messages": [response.content]}
    except Exception as e:
        return {"messages": [f"Error generating advice: {e}"]}

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