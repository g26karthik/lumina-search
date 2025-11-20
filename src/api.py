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
    doc_id: str
    score: float
    preview: str
    explanation: Dict[str, Any]

class SearchResponse(BaseModel):
    results: List[SearchResult]

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    if not search_engine:
        raise HTTPException(status_code=500, detail="Search engine not initialized")
    
    # Run CPU-bound search in thread pool
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        executor, 
        search_engine.search, 
        request.query, 
        request.top_k
    )
    
    return {"results": results}

if __name__ == "__main__":
    uvicorn.run("src.api:app", host=Config.API_HOST, port=Config.API_PORT, reload=True)
