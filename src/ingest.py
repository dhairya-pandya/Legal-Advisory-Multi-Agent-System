from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
try:
    from config import GOOGLE_API_KEY, QDRANT_URL, QDRANT_API_KEY, EMBEDDING_MODEL
except ImportError:
    from src.config import GOOGLE_API_KEY, QDRANT_URL, QDRANT_API_KEY, EMBEDDING_MODEL

def load_data():
    # FIX: Force HTTP here too
    client = QdrantClient(
        url=QDRANT_URL, 
        api_key=QDRANT_API_KEY,
        prefer_grpc=False
    )
    
    encoder = SentenceTransformer(EMBEDDING_MODEL)

    # Sample Data
    laws = [
        {"section": "IPC 378", "text": "Theft. Whoever, intending to take dishonestly any movable property out of the possession of any person without that person's consent, moves that property in order to such taking, is said to commit theft."},
        {"section": "IPC 420", "text": "Cheating and dishonestly inducing delivery of property. Whoever cheats and thereby dishonestly induces the person deceived to deliver any property to any person..."},
        {"section": "Section 138 NI Act", "text": "Dishonour of cheque for insufficiency, etc., of funds in the account. Where any cheque drawn by a person on an account maintained by him with a banker for payment of any amount of money... is returned by the bank unpaid."},
        {"section": "Consumer Protection Act", "text": "Rights of consumers include the right to be protected against the marketing of goods, products or services which are hazardous to life and property."}
    ]

    points = []
    print("⏳ Embedding laws (this might take a moment)...")
    for idx, doc in enumerate(laws):
        vector = encoder.encode(doc['text']).tolist()
        points.append(models.PointStruct(
            id=idx,
            vector=vector,
            payload=doc
        ))

    client.upsert(collection_name="legal_knowledge", points=points)
    print(f"✅ Successfully ingested {len(laws)} legal sections.")

if __name__ == "__main__":
    load_data()