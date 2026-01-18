import os
import sys
from dotenv import load_dotenv

# Load .env file
load_dotenv()

def get_env_variable(var_name):
    value = os.getenv(var_name)
    if not value:
        # Check if it's optional (like Qdrant API key for localhost)
        if var_name == "QDRANT_API_KEY" and "localhost" in os.getenv("QDRANT_URL", ""):
            return None
        print(f"❌ CRITICAL ERROR: {var_name} is missing from .env file.")
        sys.exit(1)
    return value

# Required
GOOGLE_API_KEY = get_env_variable("GOOGLE_API_KEY")

# Defaults (Safe for local)
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"