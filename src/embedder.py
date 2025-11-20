import hashlib
import re
import logging
from typing import Optional, List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from src.cache_manager import CacheManager
from src.config import Config

# Configure logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class Embedder:
    def __init__(self, model_name: str = Config.EMBEDDING_MODEL, cache_manager: Optional[CacheManager] = None):
        logger.info(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.cache_manager = cache_manager
        
        # Stats
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_time_saved_ms = 0.0
        # Approximate time per embedding (can be calibrated, but 50ms is a reasonable estimate for CPU)
        self.estimated_embedding_time_ms = 50.0 
        
        logger.info("Model loaded.")

    def log_cache_stats(self):
        if Config.ENABLE_CACHE_LOGGING:
            logger.info(f"Embedding cache summary: {self.cache_hits} hits, {self.cache_misses} misses, {round(self.total_time_saved_ms / 1000, 2)} seconds saved")

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def compute_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def embed(self, text: str, doc_id: Optional[str] = None) -> np.ndarray:
        cleaned_text = self.clean_text(text)
        
        if doc_id and self.cache_manager:
            text_hash = self.compute_hash(cleaned_text)
            cached_embedding = self.cache_manager.get_embedding(doc_id, text_hash)
            if cached_embedding is not None:
                return cached_embedding
        
        embedding = self.model.encode(cleaned_text)
        
        if doc_id and self.cache_manager:
            text_hash = self.compute_hash(cleaned_text)
            self.cache_manager.save_embedding(doc_id, text_hash, embedding)
            
        return embedding

    def embed_batch(self, texts: List[str], doc_ids: List[str]) -> np.ndarray:
        """
        Efficiently embed a batch of texts, utilizing cache where possible.
        """
        if not texts:
            return np.array([])
            
        cleaned_texts = [self.clean_text(t) for t in texts]
        embeddings = [None] * len(texts)
        indices_to_compute = []
        texts_to_compute = []
        
        # Check cache first
        if self.cache_manager:
            for i, (text, doc_id) in enumerate(zip(cleaned_texts, doc_ids)):
                text_hash = self.compute_hash(text)
                cached = self.cache_manager.get_embedding(doc_id, text_hash)
                if cached is not None:
                    embeddings[i] = cached
                    self.cache_hits += 1
                    self.total_time_saved_ms += self.estimated_embedding_time_ms
                    if Config.ENABLE_CACHE_LOGGING:
                        logger.debug(f"[Cache] HIT | {doc_id} | reused embedding | 0 ms")
                else:
                    indices_to_compute.append(i)
                    texts_to_compute.append(text)
        else:
            indices_to_compute = list(range(len(texts)))
            texts_to_compute = cleaned_texts
            
        # Compute missing embeddings in batch
        if texts_to_compute:
            logger.info(f"Computing embeddings for {len(texts_to_compute)} documents...")
            import time
            t0 = time.time()
            computed_embeddings = self.model.encode(texts_to_compute)
            t1 = time.time()
            avg_time = (t1 - t0) * 1000 / len(texts_to_compute) if texts_to_compute else 0
            
            for i, idx in enumerate(indices_to_compute):
                emb = computed_embeddings[i]
                embeddings[idx] = emb
                self.cache_misses += 1
                
                if Config.ENABLE_CACHE_LOGGING:
                    logger.debug(f"[Cache] MISS | {doc_ids[idx]} | regenerated embedding | {round(avg_time, 2)} ms")
                
                # Save to cache
                if self.cache_manager:
                    text_hash = self.compute_hash(texts_to_compute[i])
                    self.cache_manager.save_embedding(doc_ids[idx], text_hash, emb)
        
        return np.array(embeddings)
