from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from src.search_engine import SearchEngine
from src.config import Config
from src.cache_manager import CacheManager
from src.embedder import Embedder
import uvicorn
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="Multi-document Search Engine")

# Global instances
search_engine = None
executor = ThreadPoolExecutor(max_workers=4)

@app.on_event("startup")
async def startup_event():
    global search_engine
    # Initialize components with Config
    cache_manager = CacheManager(db_path=Config.CACHE_DB_PATH)
    embedder = Embedder(model_name=Config.EMBEDDING_MODEL, cache_manager=cache_manager)
    search_engine = SearchEngine(data_dir=Config.DATA_DIR, embedder=embedder)
    
    # Build index in background to not block startup completely (optional, but good for large datasets)
    # For now, we keep it blocking to ensure readiness
    search_engine.build_index()

class SearchRequest(BaseModel):
    query: str
    top_k: int = Config.DEFAULT_TOP_K

class SearchResult(BaseModel):
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from src.search_engine import SearchEngine
from src.config import Config
from src.cache_manager import CacheManager
from src.embedder import Embedder
import uvicorn
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-document Search Engine")

# Global instances
search_engine = None
executor = ThreadPoolExecutor(max_workers=4)

@app.on_event("startup")
async def startup_event():
    global search_engine
    # Initialize components with Config
    cache_manager = CacheManager(db_path=Config.CACHE_DB_PATH)
    embedder = Embedder(model_name=Config.EMBEDDING_MODEL, cache_manager=cache_manager)
    search_engine = SearchEngine(data_dir=Config.DATA_DIR, embedder=embedder)
    
    # Build index in background to not block startup completely (optional, but good for large datasets)
    # For now, we keep it blocking to ensure readiness
    search_engine.build_index()

class SearchRequest(BaseModel):
    query: str
    top_k: int = Config.DEFAULT_TOP_K

class SearchResult(BaseModel):
    doc_id: str
    score: float
    preview: str
    explanation: Dict[str, Any]

class SearchResponse(BaseModel):
    results: list
    debug: Optional[dict] = None

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    try:
        if not search_engine:
            raise HTTPException(status_code=500, detail="Search engine not initialized")
        # Run search in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        results, metrics = await loop.run_in_executor(
            None, 
            search_engine.search, 
            request.query, 
            request.top_k
        )
        
        response = {"results": results}
        if Config.ENABLE_DEBUG_METRICS:
            response["debug"] = metrics
            
        return response
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("src.api:app", host=Config.API_HOST, port=Config.API_PORT, reload=True)
