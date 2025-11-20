import streamlit as st
import requests
import time
from src.config import Config

# Page Config
st.set_page_config(
    page_title="Lumina Search",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for "Real Search Engine" Look
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #ffffff;
        color: #202124;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Centered Search Container */
    .search-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding-top: 10vh;
        padding-bottom: 2rem;
    }
    
    /* Logo Title */
    .logo-title {
        font-family: 'Product Sans', sans-serif;
        font-size: 4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4285F4, #DB4437, #F4B400, #0F9D58);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    /* Search Input Styling */
    .stTextInput > div > div > input {
        border-radius: 24px;
        border: 1px solid #dfe1e5;
        padding: 12px 24px;
        font-size: 16px;
        box-shadow: 0 1px 6px rgba(32,33,36,.28);
        transition: all 0.3s;
    }
    .stTextInput > div > div > input:hover, .stTextInput > div > div > input:focus {
        box-shadow: 0 1px 6px rgba(32,33,36,.28);
        border-color: transparent;
        background-color: #fff;
    }
    
    /* Result Card */
    .result-card {
        background-color: #fff;
        padding: 16px;
        margin-bottom: 16px;
        border-radius: 8px;
        transition: transform 0.2s;
    }
    
    /* Result Title */
    .result-title {
        color: #1a0dab;
        font-size: 20px;
        text-decoration: none;
        font-weight: 400;
        display: block;
        margin-bottom: 4px;
    }
    .result-title:hover {
        text-decoration: underline;
    }
    
    /* Result URL/ID */
    .result-meta {
        color: #006621;
        font-size: 14px;
        margin-bottom: 4px;
    }
    
    /* Result Snippet */
    .result-snippet {
        color: #4d5156;
        font-size: 14px;
        line-height: 1.58;
    }
    
    /* Explanation Box */
    .explanation-box {
        margin-top: 8px;
        padding: 8px 12px;
        background-color: #f8f9fa;
        border-left: 4px solid #4285F4;
        border-radius: 4px;
        font-size: 13px;
        color: #3c4043;
    }
    
    /* Tags */
    .keyword-tag {
        display: inline-block;
        background-color: #e8f0fe;
        color: #1967d2;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 12px;
        margin-right: 4px;
        margin-top: 4px;
    }
    
    /* Stats */
    .search-stats {
        color: #70757a;
        font-size: 14px;
        margin-bottom: 20px;
        padding-left: 16px;
    }
</style>
""", unsafe_allow_html=True)

# API Endpoint
API_URL = f"http://{Config.API_HOST}:{Config.API_PORT}/search"

# Session State
if 'query' not in st.session_state:
    st.session_state.query = ""

# --- Layout ---

# 1. Header & Search (Centered)
st.markdown('<div class="search-container"><div class="logo-title">Lumina</div></div>', unsafe_allow_html=True)

query = st.text_input("", placeholder="Search anything...", value=st.session_state.query)

# Search Button (Hidden visually but triggers submit on enter)
if query:
    st.session_state.query = query
    
    # 2. Perform Search
    try:
        start_time = time.time()
        response = requests.post(API_URL, json={"query": query, "top_k": 5})
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            
            # 3. Display Results
            st.markdown(f'<div class="search-stats">About {len(results)} results ({round(end_time - start_time, 2)} seconds)</div>', unsafe_allow_html=True)
            
            for res in results:
                # Extract Data
                doc_id = res['doc_id']
                score = res['score']
                preview = res['preview']
                explanation = res['explanation']
                
                # Explanation Data
                reason = explanation.get('reason', 'N/A')
                keywords = explanation.get('matched_keywords', [])
                vec_score = explanation.get('vector_score', 'N/A')
                bm25_score = explanation.get('bm25_score', 'N/A')
                
                keywords_html = "".join([f'<span class="keyword-tag">{k}</span>' for k in keywords])
                
                # Render Result Card
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-meta">
                        doc_id: {doc_id} &bull; Hybrid Score: {score} (Vec: {vec_score}, BM25: {bm25_score})
                    </div>
                    <a href="#" class="result-title">{doc_id} - Document Content</a>
                    <div class="result-snippet">{preview}</div>
                    
                    <div class="explanation-box">
                        <strong>💡 Insight:</strong> {reason}<br>
                        <div style="margin-top: 4px;">{keywords_html}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        st.error(f"Connection Error: {e}. Is the API running?")

else:
    # Empty state / Helper
    st.markdown("""
    <div style="text-align: center; color: #70757a; margin-top: 20px;">
        Try searching for <b>"space exploration"</b>, <b>"medical research"</b>, or <b>"computer graphics"</b>.
    </div>
    """, unsafe_allow_html=True)
