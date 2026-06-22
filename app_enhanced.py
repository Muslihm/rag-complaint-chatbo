"""
Enhanced Streamlit App with Chat History and Export
"""

import streamlit as st
import sys
import os
import json
from datetime import datetime
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'notebooks'))

try:
    from rag_pipeline import RAGPipeline
except ImportError:
    from notebooks.rag_pipeline import RAGPipeline

# ============================================
# CONFIGURATION
# ============================================

st.set_page_config(
    page_title="CrediTrust Complaint Intelligence",
    page_icon="🔍",
    layout="wide"
)

# ============================================
# STYLING
# ============================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #4A5568;
        margin-bottom: 2rem;
    }
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        background-color: #F7FAFC;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .user-msg {
        background-color: #E3F2FD;
        padding: 0.8rem 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196F3;
    }
    .bot-msg {
        background-color: #F5F5F5;
        padding: 0.8rem 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
    }
    .source-box {
        background-color: #FAFAFA;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.3rem 0;
        border: 1px solid #E0E0E0;
    }
    .metric-card {
        background-color: #F0F4F8;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE
# ============================================

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'query_count' not in st.session_state:
    st.session_state.query_count = 0

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.image("https://via.placeholder.com/150x50?text=CrediTrust", use_container_width=True)
    st.markdown("---")
    
    st.markdown("### 🏷️ Filters")
    product_filter = st.selectbox(
        "Product Category",
        ["All Products", "Credit card", "Personal loan", "Savings account", "Money transfer"]
    )
    
    top_k = st.slider("Number of Sources", 1, 10, 5)
    
    st.markdown("---")
    st.markdown("### 📊 Stats")
    st.metric("Questions Asked", st.session_state.query_count)
    st.metric("Messages", len(st.session_state.messages))
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = []
            st.session_state.query_count = 0
            st.rerun()
    with col2:
        if st.button("💾 Export", use_container_width=True):
            if st.session_state.messages:
                df = pd.DataFrame(st.session_state.messages)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

# ============================================
# MAIN CONTENT
# ============================================

st.markdown('<div class="main-header">🔍 CrediTrust Complaint Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask questions about customer complaints across financial products</div>', unsafe_allow_html=True)

# Initialize pipeline
@st.cache_resource
def get_pipeline():
    return RAGPipeline(top_k=5)

if st.session_state.pipeline is None:
    with st.spinner("Initializing RAG Pipeline..."):
        st.session_state.pipeline = get_pipeline()

# Display chat history
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg['role'] == 'user':
            st.markdown(f'<div class="user-msg"><strong>You:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-msg"><strong>AI:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
            if 'sources' in msg and msg['sources']:
                with st.expander("📚 Sources", expanded=False):
                    for source in msg['sources'][:3]:
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>Product:</strong> {source.get('product', 'Unknown')} | 
                            <strong>Issue:</strong> {source.get('issue', 'Unknown')} | 
                            <strong>Score:</strong> {source.get('score', 0):.3f}
                            <br><small>{source.get('snippet', '')}</small>
                        </div>
                        """, unsafe_allow_html=True)

# Input
st.markdown("---")
question = st.chat_input("Ask a question about customer complaints...")

if question:
    # Add user message
    st.session_state.messages.append({'role': 'user', 'content': question})
    st.session_state.query_count += 1
    
    # Process
    with st.spinner("🔍 Searching complaints..."):
        filter_val = None if product_filter == "All Products" else product_filter
        result = st.session_state.pipeline.query(question, product_filter=filter_val)
        
        # Add bot response
        st.session_state.messages.append({
            'role': 'assistant',
            'content': result['answer'],
            'sources': result['sources']
        })
    
    st.rerun()
