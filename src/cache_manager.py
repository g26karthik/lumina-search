import sqlite3
import numpy as np
from datetime import datetime
from typing import Optional

class CacheManager:
    def __init__(self, db_path: str = "cache.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    doc_id TEXT PRIMARY KEY,
                    hash TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def get_embedding(self, doc_id: str, text_hash: str) -> Optional[np.ndarray]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT embedding, hash FROM embeddings WHERE doc_id = ?", 
                    (doc_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    stored_embedding_blob, stored_hash = row
                    if stored_hash == text_hash:
                        # Deserialize embedding
                        return np.frombuffer(stored_embedding_blob, dtype=np.float32)
        except Exception as e:
            print(f"Cache read error: {e}")
            
        return None

    def save_embedding(self, doc_id: str, text_hash: str, embedding: np.ndarray):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Serialize embedding
                embedding_blob = embedding.astype(np.float32).tobytes()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO embeddings (doc_id, hash, embedding, updated_at)
                    VALUES (?, ?, ?, ?)
                """, (doc_id, text_hash, embedding_blob, datetime.now()))
                conn.commit()
        except Exception as e:
            print(f"Cache write error: {e}")
