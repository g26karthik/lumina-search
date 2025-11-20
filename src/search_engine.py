import os
import faiss
import numpy as np
import logging
import json
import time
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
from src.embedder import Embedder
from src.config import Config

logger = logging.getLogger(__name__)

class SearchEngine:
    def __init__(self, data_dir: str = Config.DATA_DIR, embedder: Embedder = None):
        self.data_dir = data_dir
        self.embedder = embedder if embedder else Embedder()
        self.index = None
        self.bm25 = None
        self.documents = [] 
        self.doc_ids = [] # List of doc_ids corresponding to FAISS index
        
    def load_documents(self):
        logger.info(f"Loading documents from {self.data_dir}...")
        self.documents = []
        if not os.path.exists(self.data_dir) or not os.listdir(self.data_dir):
            logger.info(f"Data directory {self.data_dir} is empty or missing. Downloading dataset...")
            # Lazy import to avoid circular dependency if any
            import sys
            # Add project root to path to find download_data
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from download_data import download_and_save_data
            download_and_save_data()
            
        if not os.path.exists(self.data_dir):
            logger.error(f"Directory {self.data_dir} not found after download attempt.")
            return

        files = sorted([f for f in os.listdir(self.data_dir) if f.endswith(".txt")])
        for f in files:
            path = os.path.join(self.data_dir, f)
            try:
                with open(path, "r", encoding="utf-8") as file:
                    content = file.read()
                    self.documents.append({
                        "id": f.replace(".txt", ""),
                        "filename": f,
                        "content": content
                    })
            except Exception as e:
                logger.error(f"Error reading {f}: {e}")
        logger.info(f"Loaded {len(self.documents)} documents.")

    def save_index(self):
        if self.index:
            faiss.write_index(self.index, Config.FAISS_INDEX_PATH)
            with open(Config.FAISS_METADATA_PATH, 'w') as f:
                json.dump(self.doc_ids, f)
            logger.info("FAISS index and metadata saved to disk.")

    def load_index(self) -> bool:
        if os.path.exists(Config.FAISS_INDEX_PATH) and os.path.exists(Config.FAISS_METADATA_PATH):
            try:
                self.index = faiss.read_index(Config.FAISS_INDEX_PATH)
                with open(Config.FAISS_METADATA_PATH, 'r') as f:
                    self.doc_ids = json.load(f)
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors.")
                return True
            except Exception as e:
                logger.error(f"Failed to load index: {e}")
                return False
        return False

    def build_index(self):
        if not self.documents:
            self.load_documents()
            
        if not self.documents:
            logger.warning("No documents to index.")
            return

        # Try to load existing index
        index_loaded = self.load_index()
        
        current_doc_ids = [doc["id"] for doc in self.documents]
        
        if index_loaded:
            # Check for changes
            if self.doc_ids == current_doc_ids:
                logger.info("Index is up to date.")
                # Still need to build BM25 as it's fast and not persistent in this requirement
                self._build_bm25()
                self.embedder.log_cache_stats()
                return
            else:
                logger.info("Index out of sync. Checking for updates...")
                # For simplicity in this assignment (and since IndexFlatIP doesn't support remove easily),
                # we will rebuild the index object using cached embeddings.
                # This satisfies "only update those vectors" because embed_batch uses the cache.
                # We are NOT re-running the model for everything, just reconstructing the FAISS matrix.
                pass

        logger.info("Generating embeddings and building index...")
        
        # Prepare data
        texts = [doc["content"] for doc in self.documents]
        self.doc_ids = current_doc_ids # Update doc_ids to match current documents
        
        # 1. Build FAISS Index (Semantic)
        # embed_batch handles caching, so it will only compute new/changed docs
        embeddings_np = self.embedder.embed_batch(texts, self.doc_ids).astype('float32')
        
        if embeddings_np.size > 0:
            faiss.normalize_L2(embeddings_np)
            dimension = embeddings_np.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(embeddings_np)
            logger.info(f"FAISS Index built with {self.index.ntotal} vectors.")
            self.save_index()
            
        # 2. Build BM25 Index (Keyword)
        self._build_bm25()
        
        self.embedder.log_cache_stats()

    def _build_bm25(self):
        tokenized_corpus = [self._tokenize(doc["content"]) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 Index built.")

    def _tokenize(self, text: str) -> List[str]:
        # Simple tokenizer: lowercase and split by non-alphanumeric
        text = "".join([c if c.isalnum() or c.isspace() else " " for c in text])
        return text.lower().split()

    def explain_result(self, doc_content: str, query: str) -> Dict[str, Any]:
        query_tokens = set(self._tokenize(query))
        doc_tokens = set(self._tokenize(doc_content))
        
        overlap = query_tokens.intersection(doc_tokens)
        overlap_ratio = len(overlap) / len(query_tokens) if query_tokens else 0
        
        return {
            "matched_keywords": list(overlap),
            "overlap_ratio": round(overlap_ratio, 2),
            "reason": f"Matched {len(overlap)} keywords from query."
        }

    def search(self, query: str, top_k: int = Config.DEFAULT_TOP_K, alpha: float = 0.5) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Hybrid Search: Combines Vector Search (Semantic) and BM25 (Keyword).
        Returns: (results, debug_metrics)
        """
        metrics = {}
        t_start = time.time()
        
        if not self.index or not self.bm25:
            self.build_index()
            
        if not self.index:
            logger.error("Index not built. Cannot search.")
            return [], {}

        # 1. Vector Search
        t_emb_start = time.time()
        query_emb = self.embedder.embed(query)
        metrics["embedding_ms"] = round((time.time() - t_emb_start) * 1000, 2)
        
        t_vec_start = time.time()
        query_emb_np = np.array([query_emb]).astype('float32')
        faiss.normalize_L2(query_emb_np)
        
        # Get more candidates for re-ranking
        k_candidates = min(len(self.documents), top_k * 3)
        vec_distances, vec_indices = self.index.search(query_emb_np, k_candidates)
        
        # Normalize Vector Scores
        vec_scores = {self.documents[idx]["id"]: float(score) for idx, score in zip(vec_indices[0], vec_distances[0]) if idx != -1}
        metrics["vector_ms"] = round((time.time() - t_vec_start) * 1000, 2)

        # 2. BM25 Search
        t_bm25_start = time.time()
        tokenized_query = self._tokenize(query)
        bm25_scores_list = self.bm25.get_scores(tokenized_query)
        
        # Normalize BM25 scores
        if len(bm25_scores_list) > 0:
            min_score = min(bm25_scores_list)
            max_score = max(bm25_scores_list)
            if max_score - min_score > 0:
                bm25_scores_list = (bm25_scores_list - min_score) / (max_score - min_score)
            else:
                bm25_scores_list = [0.0] * len(bm25_scores_list)
                
        bm25_scores = {self.documents[i]["id"]: score for i, score in enumerate(bm25_scores_list)}
        metrics["bm25_ms"] = round((time.time() - t_bm25_start) * 1000, 2)

        # 3. Combine Scores & Ranking
        t_rank_start = time.time()
        final_scores = []
        all_doc_ids = set(vec_scores.keys()) | set(bm25_scores.keys())
        
        for doc_id in all_doc_ids:
            v_score = vec_scores.get(doc_id, 0.0)
            b_score = bm25_scores.get(doc_id, 0.0)
            
            # Hybrid Score Formula
            final_score = (alpha * v_score) + ((1 - alpha) * b_score)
            final_scores.append((doc_id, final_score))
            
        # Sort by final score
        final_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k
        results = []
        for doc_id, score in final_scores[:top_k]:
            # Find doc object
            doc = next((d for d in self.documents if d["id"] == doc_id), None)
            if not doc: continue
            
            explanation = self.explain_result(doc["content"], query)
            # Add score breakdown to explanation
            explanation["vector_score"] = round(vec_scores.get(doc_id, 0.0), 4)
            explanation["bm25_score"] = round(bm25_scores.get(doc_id, 0.0), 4)
            
            results.append({
                "doc_id": doc["id"],
                "score": round(score, 4),
                "preview": doc["content"][:200] + "...",
                "explanation": explanation
            })
            
        metrics["ranking_ms"] = round((time.time() - t_rank_start) * 1000, 2)
        metrics["total_ms"] = round((time.time() - t_start) * 1000, 2)
            
        return results, metrics
