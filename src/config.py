import os

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data", "docs")
    CACHE_DB_PATH = os.path.join(BASE_DIR, "cache.db")
    
    # Model
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Search
    DEFAULT_TOP_K = 5
    
    # API
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    
    # Logging
    LOG_LEVEL = "INFO"
    ENABLE_CACHE_LOGGING = True
    
    # Persistence
    FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss.index")
    FAISS_METADATA_PATH = os.path.join(BASE_DIR, "faiss_meta.json")
    
    # Debug
    ENABLE_DEBUG_METRICS = True
