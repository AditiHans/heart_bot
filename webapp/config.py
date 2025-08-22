import os

# Vector store backend
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "QDRANT").upper()

# Qdrant (free cloud) – set these in Render dashboard
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "heart_faq")

# Model to load on boot (public HF repo is easiest on Render Free)
# You can switch to your own uploaded repo later.
MODEL_PATH = os.getenv("MODEL_PATH", "google/bio_clinicalbert")

# Retrieval defaults
TOP_K = int(os.getenv("TOP_K", "4"))
FETCH_K = int(os.getenv("FETCH_K", "10"))
MODE = os.getenv("MODE", "extractive")

# Answer gating for hybrid mode
MIN_TOKENS = int(os.getenv("MIN_TOKENS", "6"))
MIN_CONF = float(os.getenv("MIN_CONF", "0.15"))

# CORS
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]