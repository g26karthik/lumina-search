import os
import faiss
import numpy as np
import logging
from typing import List, Dict, Any
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
        self.doc_ids = [] 
        
    def load_documents(self):
        logger.info(f"Loading documents from {self.data_dir}...")
        self.documents = []
        if not os.path.exists(self.data_dir):
            logger.error(f"Directory {self.data_dir} not found.")
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

    def build_index(self):
        if not self.documents:
            self.load_documents()
            
        if not self.documents:
            logger.warning("No documents to index.")
            return

        logger.info("Generating embeddings and building index...")
        
        # Prepare data
        texts = [doc["content"] for doc in self.documents]
        doc_ids = [doc["id"] for doc in self.documents]
        
        # 1. Build FAISS Index (Semantic)
        embeddings_np = self.embedder.embed_batch(texts, doc_ids).astype('float32')
        if embeddings_np.size > 0:
            faiss.normalize_L2(embeddings_np)
            dimension = embeddings_np.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(embeddings_np)
            logger.info(f"FAISS Index built with {self.index.ntotal} vectors.")
            
        # 2. Build BM25 Index (Keyword)
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

    def search(self, query: str, top_k: int = Config.DEFAULT_TOP_K, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """
        Hybrid Search: Combines Vector Search (Semantic) and BM25 (Keyword).
        alpha: Weight for Vector Search (0.0 to 1.0). 0.5 means equal weight.
        """
        if not self.index or not self.bm25:
            self.build_index()
            
        if not self.index:
            logger.error("Index not built. Cannot search.")
            return []

        # 1. Vector Search
        query_emb = self.embedder.embed(query)
        query_emb_np = np.array([query_emb]).astype('float32')
        faiss.normalize_L2(query_emb_np)
        
        # Get more candidates for re-ranking
        k_candidates = min(len(self.documents), top_k * 3)
        vec_distances, vec_indices = self.index.search(query_emb_np, k_candidates)
        
        # Normalize Vector Scores (Cosine Similarity is already -1 to 1, but usually 0-1 for text)
        # FAISS IP with normalized vectors returns Cosine Similarity
        vec_scores = {self.documents[idx]["id"]: float(score) for idx, score in zip(vec_indices[0], vec_distances[0]) if idx != -1}

        # 2. BM25 Search
        tokenized_query = self._tokenize(query)
        bm25_scores_list = self.bm25.get_scores(tokenized_query)
        
        # Normalize BM25 scores (Min-Max Normalization)
        if len(bm25_scores_list) > 0:
            min_score = min(bm25_scores_list)
            max_score = max(bm25_scores_list)
            if max_score - min_score > 0:
                bm25_scores_list = (bm25_scores_list - min_score) / (max_score - min_score)
            else:
                bm25_scores_list = [0.0] * len(bm25_scores_list)
                
        bm25_scores = {self.documents[i]["id"]: score for i, score in enumerate(bm25_scores_list)}

        # 3. Combine Scores
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
            
        return results
