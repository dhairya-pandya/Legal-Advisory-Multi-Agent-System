from qdrant_client import QdrantClient, models
from src.config import QDRANT_URL, QDRANT_API_KEY

def init_db():
    try:
        print(f"☁️ Connecting to Qdrant Cloud...")
        
        # FIX: Force HTTP (REST) mode to bypass Firewall/Timeout issues
        client = QdrantClient(
            url=QDRANT_URL, 
            api_key=QDRANT_API_KEY,
            prefer_grpc=False, 
            timeout=60
        )
        
        # 1. Legal Knowledge
        if not client.collection_exists("legal_knowledge"):
            client.create_collection(
                collection_name="legal_knowledge",
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
            )
            print("✅ Created Collection: legal_knowledge")
        else:
            print("ℹ️ Collection 'legal_knowledge' already exists.")

        # 2. Case Memory
        if not client.collection_exists("case_memory"):
            client.create_collection(
                collection_name="case_memory",
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
            )
            print("✅ Created Collection: case_memory")
        else:
             print("ℹ️ Collection 'case_memory' already exists.")

        # 3. Evidence Vault
        if not client.collection_exists("evidence_vault"):
            client.create_collection(
                collection_name="evidence_vault",
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
            )
            print("✅ Created Collection: evidence_vault")
        else:
             print("ℹ️ Collection 'evidence_vault' already exists.")

    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        print("💡 Check your VPN or Firewall settings.")

if __name__ == "__main__":
    init_db()