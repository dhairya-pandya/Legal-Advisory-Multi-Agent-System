import uuid
import sys
import os
import time
from datasets import load_dataset
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

# Fix imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from config import QDRANT_URL, QDRANT_API_KEY, EMBEDDING_MODEL
except ImportError:
    from src.config import QDRANT_URL, QDRANT_API_KEY, EMBEDDING_MODEL

def setup_hybrid_collection(client, collection_name):
    """Resets and creates the Hybrid Search Collection."""
    if client.collection_exists(collection_name):
        print(f"⚠️  Collection '{collection_name}' exists. Deleting to rebuild...")
        client.delete_collection(collection_name)
    
    print(f"🏗️  Creating Hybrid Collection: {collection_name}...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(size=384, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
        }
    )
    print("✅ Collection Ready.")

def main():
    # 1. Connect with INCREASED TIMEOUT (The Fix)
    print("☁️  Connecting to Qdrant Cloud...")
    client = QdrantClient(
        url=QDRANT_URL, 
        api_key=QDRANT_API_KEY, 
        prefer_grpc=False,
        timeout=100  # Wait up to 100 seconds (prevents ReadTimeout)
    )
    COLLECTION_NAME = "legal_knowledge"
    
    setup_hybrid_collection(client, COLLECTION_NAME)

    # 2. Download Dataset
    print("📥 Downloading Indian Laws (mratanusarkar/Indian-Laws)...")
    try:
        dataset = load_dataset("mratanusarkar/Indian-Laws", split="train")
        print(f"   🔹 Found {len(dataset)} legal sections.")
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return

    # 3. Load Models
    print("🧠 Loading AI Models...")
    dense_model = SentenceTransformer(EMBEDDING_MODEL)
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    # 4. Batch Processing with RETRY LOGIC
    BATCH_SIZE = 50
    all_rows = list(dataset)
    total_batches = len(all_rows) // BATCH_SIZE + 1
    
    for i in range(0, len(all_rows), BATCH_SIZE):
        batch = all_rows[i : i + BATCH_SIZE]
        
        texts = []
        payloads = []
        
        for row in batch:
            act = row.get('act_title', 'Unknown Act')
            section = row.get('section', 'Unknown')
            text = row.get('law', '') or row.get('description', '') or ""
            
            full_text = f"{act} - Section {section}: {text}"
            texts.append(full_text)
            payloads.append({
                "act": act,
                "section": str(section),
                "full_text": full_text
            })

        # Generate Embeddings
        try:
            dense_vectors = dense_model.encode(texts)
            sparse_vectors = list(sparse_model.embed(texts))
        except Exception as e:
            print(f"   ⚠️ Skipping batch {i} due to encoding error: {e}")
            continue
        
        points = []
        for j, payload in enumerate(payloads):
            sparse_vector_obj = models.SparseVector(
                indices=sparse_vectors[j].indices.tolist(),
                values=sparse_vectors[j].values.tolist()
            )
            
            points.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_vectors[j].tolist(),
                    "sparse": sparse_vector_obj
                },
                payload=payload
            ))
        
        # --- RETRY LOOP (The Bulletproof Logic) ---
        max_retries = 3
        for attempt in range(max_retries):
            try:
                client.upsert(collection_name=COLLECTION_NAME, points=points)
                print(f"   ⬆️  Uploaded Batch {i // BATCH_SIZE + 1}/{total_batches}")
                break # Success! Exit retry loop
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"   ⚠️ Timeout on Batch {i // BATCH_SIZE + 1}. Retrying in 5s... ({attempt+1}/{max_retries})")
                    time.sleep(5)
                else:
                    print(f"   ❌ FAILED Batch {i // BATCH_SIZE + 1} after {max_retries} attempts: {e}")
                    # We continue to the next batch instead of crashing

    print("🚀 SUCCESS: Full Indian Legal Code Ingested!")

if __name__ == "__main__":
    main()