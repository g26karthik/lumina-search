import streamlit as st
import time
import sys
import os
import textwrap

# Add project root to path so src can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.search_engine import SearchEngine

# Page Config
st.set_page_config(
    page_title="Lumina Search",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Logic: Direct Integration (No API needed for Streamlit Cloud) ---
@st.cache_resource
def get_search_engine():
    # Initialize engine directly
    engine = SearchEngine()
    engine.build_index() # This handles auto-download if needed
    return engine

try:
    search_engine = get_search_engine()
except Exception as e:
    st.error(f"Failed to initialize search engine: {e}")
    st.stop()

# --- Custom CSS ---
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Search Container */
    .search-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding-top: 8vh;
        padding-bottom: 2rem;
    }
    
    /* Logo */
    .logo-title {
        font-family: 'Product Sans', sans-serif;
        font-size: 4.5rem;
        font-weight: 700;
        letter-spacing: -1px;
        background: linear-gradient(90deg, #4285F4, #DB4437, #F4B400, #0F9D58);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    /* Subtitle */
    .subtitle {
        color: #5f6368;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    /* Search Input Styling */
    .stTextInput > div > div > input {
        border-radius: 24px;
        border: 1px solid #dfe1e5;
        padding: 12px 24px;
        font-size: 16px;
        height: 48px;
        box-shadow: 0 1px 6px rgba(32,33,36,.28);
        transition: all 0.3s;
        color: #202124;
        background-color: #ffffff; /* Force white background */
    }
    .stTextInput > div > div > input:hover, .stTextInput > div > div > input:focus {
        box-shadow: 0 2px 8px rgba(32,33,36,.35);
        border-color: transparent;
        outline: none;
        background-color: #ffffff;
    }
    
    /* Result Card */
    .result-card {
        background-color: #fff;
        padding: 0px;
        margin-bottom: 24px;
        border-radius: 8px;
        font-family: arial, sans-serif;
    }
    
    /* Result Meta (URL-like) */
    .result-meta {
        color: #202124;
        font-size: 14px;
        display: flex;
        align-items: center;
        margin-bottom: 4px;
    }
    .result-id {
        color: #202124;
        font-weight: 400;
    }
    .result-score {
        color: #5f6368;
        font-size: 12px;
        margin-left: 8px;
    }
    
    /* Result Title */
    .result-title {
        color: #1a0dab;
        font-size: 20px;
        text-decoration: none;
        font-weight: 400;
        display: block;
        margin-bottom: 4px;
        line-height: 1.3;
    }
    .result-title:hover {
        text-decoration: underline;
    }
    
    /* Snippet */
    .result-snippet {
        color: #4d5156;
        font-size: 14px;
        line-height: 1.58;
        margin-bottom: 8px;
    }
    
    /* Insight Box */
    .insight-box {
        background-color: #f8f9fa;
        border: 1px solid #dadce0;
        border-radius: 8px;
        padding: 12px;
        margin-top: 8px;
    }
    .insight-header {
        font-size: 12px;
        font-weight: 700;
        color: #5f6368;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .insight-text {
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
        font-weight: 500;
        margin-right: 4px;
        margin-top: 4px;
    }
    
    /* Stats */
    .search-stats {
        color: #70757a;
        font-size: 14px;
        margin-bottom: 20px;
        padding-left: 0px;
    }
</style>
""", unsafe_allow_html=True)

# --- Layout ---

# Header
st.markdown("""
    <div class="search-container">
        <div class="logo-title">Lumina</div>
        <div class="subtitle">Deep Semantic Search over 20 Newsgroups</div>
    </div>
""", unsafe_allow_html=True)

# Search Input
query = st.text_input("Search Query", placeholder="Search for quantum physics, space, medicine...", value=st.session_state.get("query", ""), label_visibility="collapsed")

if query:
    st.session_state.query = query
    
    start_time = time.time()
    # Direct call to search engine
    results, metrics = search_engine.search(query, top_k=5)
    end_time = time.time()
    
    # Stats
    st.markdown(f'<div class="search-stats">About {len(results)} results ({round(end_time - start_time, 3)} seconds)</div>', unsafe_allow_html=True)
    
    # Debug Metrics
    if Config.ENABLE_DEBUG_METRICS and metrics:
        with st.expander("Debug Metrics (Latency)"):
            st.json(metrics)
    
    for res in results:
        doc_id = res['doc_id']
        score = res['score']
        preview = res['preview']
        explanation = res['explanation']
        
        reason = explanation.get('reason', 'N/A')
        keywords = explanation.get('matched_keywords', [])
        vec_score = explanation.get('vector_score', 'N/A')
        bm25_score = explanation.get('bm25_score', 'N/A')
        
        keywords_html = "".join([f'<span class="keyword-tag">{k}</span>' for k in keywords])
        
        # Use textwrap.dedent to prevent indentation from being interpreted as code blocks
        # IMPORTANT: The HTML must be flat (no indentation) to avoid Markdown code block rendering
        html_content = textwrap.dedent(f"""
        <div class="result-card">
        <a href="#" class="result-title" style="font-size: 18px; text-decoration: none; color: #1a0dab;">Document Content Preview for doc_id : {doc_id},</a>
        <div style="margin-top: 6px; font-family: monospace; font-size: 14px; color: #202124;">
        "score": {score} <span style="color: #5f6368;">(Vec: {vec_score} | BM25: {bm25_score})</span> ,
        </div>
        <div class="insight-box" style="margin: 12px 0;">
        <div class="insight-header">Why this matched</div>
        <div class="insight-text">
        {reason}<br>
        <div style="margin-top: 6px;">{keywords_html}</div>
        </div>
        </div>
        <div style="font-family: monospace; font-size: 14px; color: #4d5156; line-height: 1.5;">
        "preview": "{preview}"
        </div>
        </div>
        <hr style="margin: 20px 0; border: 0; border-top: 1px solid #dadce0;">
        """)
        
        st.markdown(html_content, unsafe_allow_html=True)

