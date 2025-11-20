from src.search_engine import SearchEngine
import time

def verify():
    print("Initializing Search Engine...")
    start = time.time()
    engine = SearchEngine()
    engine.build_index()
    print(f"Initialization took {time.time() - start:.2f}s")
    
    query = "space exploration"
    print(f"\nSearching for: '{query}'")
    results = engine.search(query, top_k=3)
    
    for res in results:
        print(f"\nDoc ID: {res['doc_id']}")
        print(f"Score: {res['score']}")
        print(f"Explanation: {res['explanation']}")
        print("-" * 30)

if __name__ == "__main__":
    verify()
