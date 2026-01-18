import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# 1. Force reload the .env file
load_dotenv(override=True)

# 2. Get variables
url = os.getenv("QDRANT_URL")
key = os.getenv("QDRANT_API_KEY")

print(f"🔎 DEBUG: The script thinks QDRANT_URL is: [{url}]")
print(f"🔎 DEBUG: The API Key length is: {len(key) if key else '0'} characters")

# 3. Try to connect
if not url or "localhost" in url:
    print("❌ ERROR: You are still pointing to LOCALHOST. The .env file was not updated or saved.")
else:
    print("☁️ Attempting to connect to Cloud...")
    try:
        client = QdrantClient(url=url, api_key=key, timeout=10)
        collections = client.get_collections()
        print(f"✅ SUCCESS! Connected to Qdrant Cloud. Found {len(collections.collections)} collections.")
    except Exception as e:
        print(f"❌ CONNECTION FAILED: {e}")