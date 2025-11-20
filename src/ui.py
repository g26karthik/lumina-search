import streamlit as st
import requests
import json

# Page Config
st.set_page_config(
    page_title="Lumina Search", 
    page_icon="🔍", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #ffffff;
    }
    
    /* Search Input */
    .stTextInput > div > div > input {
        background-color: #21262d;
        color: #ffffff;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 10px;
    }
    .stTextInput > div > div > input:focus {
        border-color: #58a6ff;
        box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.3);
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #238636;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background-color: #2ea043;
        border-color: #2ea043;
    }
    
    /* Result Cards */
    .result-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: transform 0.2s, border-color 0.2s;
    }
    .result-card:hover {
        border-color: #58a6ff;
        transform: translateY(-2px);
    }
    .result-score {
        font-size: 0.85rem;
        color: #8b949e;
        margin-bottom: 0.5rem;
    }
    .result-preview {
        font-size: 1rem;
        line-height: 1.5;
        color: #c9d1d9;
        margin-bottom: 1rem;
    }
    .explanation-box {
        background-color: #21262d;
        border-radius: 6px;
        padding: 0.75rem;
        font-size: 0.9rem;
        border-left: 3px solid #a371f7;
    }
    .keyword-tag {
        background-color: rgba(56, 139, 253, 0.15);
        color: #58a6ff;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.8rem;
        margin-right: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("---")
    api_url = st.text_input("API Endpoint", "http://localhost:8000/search")
    top_k = st.slider("Results per search", 1, 20, 5)
    
    st.markdown("### About")
    st.info(
        "This engine uses **Sentence Transformers** for semantic embedding and **FAISS** for vector search. "
        "It includes a custom caching layer to optimize performance."
    )

# Main Content
st.title("Lumina Search")
st.markdown("### Explore your document knowledge base")

# Search Section
col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_input("", placeholder="Ask a question or search for a topic...", label_visibility="collapsed")
with col2:
    search_button = st.button("Search", use_container_width=True)

if search_button or query:
    if not query:
        st.warning("Please enter a search query.")
    else:
        with st.spinner("Searching the cosmos..."):
            try:
                response = requests.post(api_url, json={"query": query, "top_k": top_k})
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    
                    st.markdown(f"Found **{len(results)}** relevant documents")
                    st.markdown("---")
                    
                    for res in results:
                        # Format explanation keywords
                        keywords = res['explanation'].get('matched_keywords', [])
                        keywords_html = "".join([f'<span class="keyword-tag">{k}</span>' for k in keywords])
                        
                        # Score breakdown
                        vec_score = res['explanation'].get('vector_score', 'N/A')
                        bm25_score = res['explanation'].get('bm25_score', 'N/A')
                        
                        st.markdown(f"""
                        <div class="result-card">
                            <div class="result-score">
                                Hybrid Score: {res['score']} | Vector: {vec_score} | BM25: {bm25_score} | Doc ID: {res['doc_id']}
                            </div>
                            <div class="result-preview">{res['preview']}</div>
                            <div class="explanation-box">
                                <strong>Why this matched:</strong> {res['explanation'].get('reason', 'N/A')}<br>
                                <div style="margin-top: 8px;">{keywords_html}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"Connection failed: {e}. Is the API running?")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #8b949e; font-size: 0.8rem;'>"
    "Built by G Karthik Koundinya with FastAPI & Streamlit"
    "</div>", 
    unsafe_allow_html=True
)
