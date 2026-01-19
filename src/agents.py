import os
import sys
import streamlit as st
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder

# --- 1. CONFIG SETUP ---
# Ensure we can import config.py whether running locally or on Cloud
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from config import GOOGLE_API_KEY, QDRANT_URL, QDRANT_API_KEY, EMBEDDING_MODEL
except ImportError:
    from src.config import GOOGLE_API_KEY, QDRANT_URL, QDRANT_API_KEY, EMBEDDING_MODEL

# --- 2. CACHED TOOLS (AI Brain) ---
@st.cache_resource
def get_ai_tools():
    # 1. LLM: Gemini 1.5 Flash with Safety Filters DISABLED
    # This allows the AI to discuss "crime", "murder", "punishment" without getting blocked.
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
    
    # 2. Database: Qdrant Client (High Timeout prevents network errors)
    client = QdrantClient(
        url=QDRANT_URL, 
        api_key=QDRANT_API_KEY, 
        prefer_grpc=False,
        timeout=60
    )
    
    # 3. Retriever: Fast Embedding Model (Finds ~15 potential matches)
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    
    # 4. Reranker: Precision Model (Filters down to top 3-5 actual matches)
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2') 
    
    return llm, client, encoder, reranker

# Initialize the tools once
llm, client, encoder, reranker = get_ai_tools()

# --- 3. AGENT STATE DEFINITION ---
class AgentState(TypedDict):
    messages: List[str]
    current_agent: str
    context_data: str

# --- 4. AGENT FUNCTIONS ---

def intake_router(state: AgentState):
    """
    Agent A: The Receptionist.
    Decides if the user needs a Lawyer (Legal Clerk) or an Auditor (Evidence).
    """
    if not state['messages']:
        return {"current_agent": "senior_counsel"}
    last_msg = state['messages'][-1].lower()
    
    # Keywords to route to Evidence Auditor
    if any(x in last_msg for x in ["photo", "image", "scan", "contract", "upload", "document"]):
        return {"current_agent": "evidence_auditor"}
    # Keywords to route to Legal Clerk
    elif any(x in last_msg for x in ["law", "section", "act", "punishment", "ipc", "rights", "crime", "police", "court", "arrest"]):
        return {"current_agent": "legal_clerk"}
    else:
        # Default to Senior Counsel for general advice
        return {"current_agent": "senior_counsel"}

def legal_clerk(state: AgentState):
    """
    Agent B: The Researcher.
    Uses Hybrid Search + Reranking to find specific laws.
    """
    query = state['messages'][-1]
    
    try:
        # Step 1: Broad Retrieval (Get top 15 candidates)
        query_vector = encoder.encode(query).tolist()
        
        hits = client.query_points(
            collection_name="legal_knowledge",
            query=query_vector,
            using="dense", # Search using the semantic vector
            limit=15, 
            with_payload=True
        ).points
        
        if not hits:
            return {"context_data": "No matches found.", "messages": ["I searched the legal database but found no relevant laws."]}

        # Step 2: Reranking (Precision Filtering)
        # Extract text from hits
        docs = [hit.payload.get('full_text', hit.payload.get('text', '')) for hit in hits]
        
        # Create pairs [("Query", "Law 1"), ("Query", "Law 2")...]
        pairs = [[query, doc] for doc in docs]
        
        # Get relevance scores
        scores = reranker.predict(pairs)
        
        # Sort results by score (Highest first)
        ranked_hits = sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)
        
        # Step 3: Selection (Keep top 5 valid results)
        final_results = []
        for score, hit in ranked_hits[:5]:
            if score > 0.01: # Filter out complete noise
                text = hit.payload.get('full_text', hit.payload.get('text', ''))
                final_results.append(f"- [Relevance: {score:.2f}] {text}")
        
        if not final_results:
             return {"context_data": "Laws found but low relevance.", "messages": ["I found some laws, but they didn't seem relevant to your specific question."]}
            
        return {
            "context_data": "\n".join(final_results), 
            "messages": [f"I have identified {len(final_results)} highly relevant legal precedents for your case."]
        }

    except Exception as e:
        # Fallback if database fails
        return {"context_data": f"Database Error: {e}", "messages": ["I am momentarily unable to access the legal archives."]}

def evidence_auditor(state: AgentState):
    """
    Agent C: The Forensic Analyst.
    (Currently a placeholder for the Vision/Document analysis feature).
    """
    return {
        "context_data": "Document Scan: Valid Rent Agreement. Missing witness signature on Page 2.", 
        "messages": ["I have analyzed the document. It appears to be a valid Rent Agreement, but I noticed it is missing a witness signature."]
    }

def senior_counsel(state: AgentState):
    """
    Agent D: The Senior Lawyer.
    Synthesizes facts and laws into actionable advice.
    """
    history = state['messages']
    context = state.get('context_data', 'No specific context provided.')
    
    prompt = f"""
    You are 'Justitia', an AI Legal Co-Counsel for India.
    
    USER QUERY HISTORY: {history}
    LEGAL EVIDENCE FROM DATABASE: {context}
    
    INSTRUCTIONS:
    1. Answer the user's question using ONLY the provided Legal Evidence.
    2. Cite specific Acts, Sections, and punishments mentioned in the evidence.
    3. If the evidence is missing or irrelevant, provide general legal principles but explicitly warn the user.
    4. Be professional, empathetic, and concise.
    """
    
    try:
        response = llm.invoke(prompt)
        return {"messages": [response.content]}
    except Exception as e:
        return {"messages": [f"I encountered an error generating your advice. Please try rephrasing your question. (Error: {str(e)})"]}

# --- 5. WORKFLOW CONSTRUCTION (LangGraph) ---
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("intake", intake_router)
workflow.add_node("legal_clerk", legal_clerk)
workflow.add_node("evidence_auditor", evidence_auditor)
workflow.add_node("senior_counsel", senior_counsel)

# Add Edges
workflow.set_entry_point("intake")

workflow.add_conditional_edges(
    "intake",
    lambda x: x['current_agent'],
    {
        "legal_clerk": "legal_clerk", 
        "evidence_auditor": "evidence_auditor", 
        "senior_counsel": "senior_counsel"
    }
)

workflow.add_edge("legal_clerk", "senior_counsel")
workflow.add_edge("evidence_auditor", "senior_counsel")
workflow.add_edge("senior_counsel", END)

# Compile the graph
app = workflow.compile()