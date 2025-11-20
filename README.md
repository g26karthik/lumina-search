# Multi-document Embedding Search Engine with Caching

> **Upgrade Summary:**
> 🔹 **Persistent FAISS Index** — Indexing is incremental; `faiss.index` is saved/loaded to avoid rebuilding.
> 🔹 **Smart Caching** — Logs cache hits/misses and calculates real-time savings (e.g., "78.4 seconds saved").
> 🔹 **Query Latency Metrics** — Search API returns detailed breakdown (embedding, vector search, BM25, ranking) for profiling.

**Project Name**: Multi-document Embedding Search Engine with Caching
**Author**: G Karthik

A lightweight, efficient search engine built over the 20 Newsgroups dataset. It features **Hybrid Search** (Vector + BM25), **Local Caching**, **Async API**, and a **Streamlit UI**.

---

## 📂 Folder Structure

```
/
├── data/               # Stores text files (Ignored by Git)
│   └── docs/           # Document collection
├── src/
│   ├── __init__.py
│   ├── config.py       # Centralized configuration
│   ├── cache_manager.py # SQLite caching implementation
│   ├── embedder.py      # Embedding generation (Batching + Caching)
│   ├── search_engine.py # Hybrid Search (FAISS + BM25) & Ranking Logic
│   ├── api.py           # FastAPI application (Async)
│   └── ui.py            # Streamlit User Interface
├── requirements.txt    # Project dependencies
├── README.md           # Documentation
├── download_data.py    # Script to download dataset
└── verify.py           # Script to evaluate retrieval quality
```

---

## 🚀 How to Run

### 1. Setup
Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Data Preparation
Download the 20 Newsgroups dataset:
### 4. Start the API
Run the FastAPI server:
```bash
uvicorn src.api:app --reload
```
- **Endpoint**: `POST /search`
- **Docs**: `http://127.0.0.1:8000/docs`

### 5. Start the UI (Bonus)
Run the Streamlit interface:
```bash
streamlit run src/ui.py
```
- Access at: `http://localhost:8501`

---

## 🧠 Design Choices

### A. How Caching Works
Implemented a **SQLite** based `CacheManager` (`src/cache_manager.py`).
- **Schema**: `(doc_id TEXT PRIMARY KEY, hash TEXT, embedding BLOB, updated_at TIMESTAMP)`
- **Logic**:
    1.  Calculate `SHA256` hash of the preprocessed document text.
    2.  Check DB: If `doc_id` exists AND `hash` matches, load the BLOB.
    3.  If mismatch or missing: Generate new embedding -> Update DB.
- **Benefit**: Prevents recomputing embeddings for unchanged files, persisting across restarts.

### B. Embedding Model
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Reason**: Recommended in the assessment. It provides a perfect balance of speed and semantic accuracy for this scale (100-200 docs).
- **Optimization**: Implemented **Batch Processing** (`embed_batch`) to encode multiple documents simultaneously, significantly speeding up the initial index build.

### C. Search Algorithm (Hybrid "Superior" Logic)
Instead of relying on just one method, I implemented **Hybrid Search**:
1.  **Vector Search (FAISS)**: Uses `IndexFlatIP` with L2 normalized embeddings (equivalent to Cosine Similarity). Captures *semantic meaning* (e.g., "cosmos" matches "space").
2.  **Keyword Search (BM25)**: Uses `rank_bm25`. Captures *exact keyword matches* which vectors sometimes miss.
3.  **Combination**: Scores are normalized and combined (`0.5 * Vector + 0.5 * BM25`) for robust ranking.

### D. API Architecture
- **Framework**: FastAPI (Preferred).
- **Concurrency**: The `/search` endpoint is `async` and uses a `ThreadPoolExecutor`. This ensures that CPU-intensive search operations do not block the main event loop, allowing the API to handle multiple concurrent requests efficiently.

### E. Ranking Explanation
Satisfies the mandatory requirement by providing:
- **Why matched**: A text summary.
- **Keywords**: List of overlapping tokens between query and document.
- **Score Breakdown**: Explicitly shows Vector vs. BM25 contribution in the UI.

### F. Persistent Indexing & Incremental Updates
- **Behavior**: On the first run, the system builds the FAISS index and saves it to `faiss.index` (along with metadata in `faiss_meta.json`).
- **Incremental**: On subsequent runs, it loads the index from disk. It checks for new or changed documents (using hash comparison) and only computes embeddings for those, appending them to the index. This significantly reduces startup time.

### G. Cache Performance Logging
- **Tracking**: The system tracks every cache hit and miss.
- **Output**: On startup (and during batch operations), it logs a summary:
  ```
  Embedding cache summary: 143 hits, 12 misses, 78.4 seconds saved
  ```
- **Config**: Controlled via `ENABLE_CACHE_LOGGING = True` in `src/config.py`.

### H. Query Latency Metrics
- **Profiling**: Every search request measures the time taken for each stage of the pipeline.
- **Response**: The API includes a `debug` field with the breakdown:
  ```json
  "debug": {
    "total_ms": 73.4,
    "embedding_ms": 12.1,
    "vector_ms": 21.8,
    "bm25_ms": 26.4,
    "ranking_ms": 13.1
  }
  ```
