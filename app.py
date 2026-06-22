"""
Task 4: Interactive Chat Interface for RAG System
CrediTrust Financial - Complaint Intelligence System
"""

import streamlit as st
import sys
import os
import time
from datetime import datetime
import pandas as pd

# Add notebooks to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'notebooks'))

# Import RAG pipeline
try:
    from rag_pipeline import RAGPipeline
except ImportError:
    from notebooks.rag_pipeline import RAGPipeline

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="CrediTrust Complaint Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================

st.markdown("""
<style>
    /* Main header */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
        padding: 1rem 0;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #4A5568;
        margin-bottom: 2rem;
    }
    
    /* Chat message styling */
    .user-message {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196F3;
    }
    
    .bot-message {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
    }
    
    /* Source card styling */
    .source-card {
        background-color: #FAFAFA;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.3rem 0;
        border: 1px solid #E0E0E0;
        border-left: 4px solid #FF9800;
    }
    
    .source-product {
        font-weight: 600;
        color: #1E3A5F;
    }
    
    .source-issue {
        color: #E65100;
        font-weight: 500;
    }
    
    .source-score {
        color: #4CAF50;
        font-weight: 500;
    }
    
    .source-text {
        background-color: #F5F5F5;
        padding: 0.5rem;
        border-radius: 4px;
        font-size: 0.9rem;
        border: 1px solid #E0E0E0;
        margin-top: 0.3rem;
    }
    
    /* Sidebar styling */
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1E3A5F;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background-color: #F0F4F8;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1E3A5F;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #4A5568;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #A0AEC0;
        font-size: 0.8rem;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid #E2E8F0;
        margin-top: 2rem;
    }
    
    /* Divider */
    .divider {
        border-top: 2px solid #E2E8F0;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE INITIALIZATION
# ============================================

def init_session_state():
    """Initialize all session state variables"""
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    
    if 'total_queries' not in st.session_state:
        st.session_state.total_queries = 0
    
    if 'product_filter' not in st.session_state:
        st.session_state.product_filter = "All Products"

# ============================================
# PIPELINE INITIALIZATION
# ============================================

@st.cache_resource
def init_pipeline():
    """Initialize the RAG pipeline with caching"""
    try:
        with st.spinner("Initializing RAG Pipeline..."):
            pipeline = RAGPipeline(top_k=5)
            st.session_state.initialized = True
            return pipeline
    except Exception as e:
        st.error(f"Failed to initialize RAG pipeline: {e}")
        return None

# ============================================
# SIDEBAR
# ============================================

def render_sidebar():
    """Render the sidebar with filters and metrics"""
    
    with st.sidebar:
        st.markdown('<div class="sidebar-header">🔍 CrediTrust</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Product Filter
        st.markdown("### 🏷️ Product Filter")
        product_filter = st.selectbox(
            "Select a product to filter complaints:",
            ["All Products", "Credit card", "Personal loan", "Savings account", "Money transfer"],
            key="product_filter_select"
        )
        
        # Apply filter button
        if st.button("Apply Filter", use_container_width=True):
            st.session_state.product_filter = product_filter
            st.rerun()
        
        st.markdown("---")
        
        # Number of sources
        st.markdown("### 📚 Retrieval Settings")
        top_k = st.slider(
            "Number of sources to retrieve:",
            min_value=1,
            max_value=10,
            value=5,
            key="top_k_slider"
        )
        
        st.markdown("---")
        
        # Statistics
        st.markdown("### 📊 Session Stats")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{st.session_state.total_queries}</div>
                <div class="metric-label">Total Questions</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(st.session_state.messages)}</div>
                <div class="metric-label">Messages</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Clear button
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.total_queries = 0
            st.rerun()
        
        st.markdown("---")
        
        # About
        with st.expander("ℹ️ About"):
            st.markdown("""
            **CrediTrust Complaint Intelligence**
            
            This system uses RAG (Retrieval-Augmented Generation) to answer questions
            about customer complaints across financial products.
            
            **Products Covered:**
            - Credit Cards
            - Personal Loans
            - Savings Accounts
            - Money Transfers
            
            **Data Source:** CFPB Complaint Database
            """)
        
        st.markdown("---")
        st.markdown('<div class="footer">v1.0 | CrediTrust Financial</div>', unsafe_allow_html=True)

# ============================================
# DISPLAY MESSAGES
# ============================================

def display_message(message):
    """Display a single message in the chat"""
    
    if message['role'] == 'user':
        with st.chat_message("user"):
            st.markdown(message['content'])
    else:
        with st.chat_message("assistant"):
            st.markdown(message['content'])
            
            # Display sources if available
            if 'sources' in message and message['sources']:
                with st.expander("📚 View Sources", expanded=False):
                    for i, source in enumerate(message['sources'][:5], 1):
                        st.markdown(f"""
                        <div class="source-card">
                            <span class="source-product">Product:</span> {source.get('product', 'Unknown')} &nbsp;|&nbsp;
                            <span class="source-issue">Issue:</span> {source.get('issue', 'Unknown')} &nbsp;|&nbsp;
                            <span class="source-score">Score:</span> {source.get('score', 0):.3f}
                            <div class="source-text">{source.get('snippet', '')}</div>
                        </div>
                        """, unsafe_allow_html=True)

def display_welcome():
    """Display welcome message"""
    
    welcome_msg = """
    ### 👋 Welcome to CrediTrust Complaint Intelligence!
    
    I can help you analyze customer complaints across our financial products.
    
    **Try asking me:**
    - "What are the most common credit card complaints?"
    - "Why are customers unhappy with money transfers?"
    - "Tell me about fraud-related complaints"
    - "Compare complaints between credit cards and personal loans"
    
    💡 **Tip:** Use the sidebar to filter by product or adjust the number of sources.
    """
    
    with st.chat_message("assistant"):
        st.markdown(welcome_msg)

# ============================================
# PROCESS QUESTION
# ============================================

def process_question(question: str, pipeline, product_filter: str, top_k: int):
    """Process a user question and generate response"""
    
    # Update pipeline top_k if changed
    pipeline.top_k = top_k
    
    # Convert filter
    filter_val = None if product_filter == "All Products" else product_filter
    
    # Show thinking indicator
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching complaints..."):
            # Query the pipeline
            result = pipeline.query(question, product_filter=filter_val)
            
            # Update statistics
            st.session_state.total_queries += 1
            
            # Add to messages
            st.session_state.messages.append({
                'role': 'assistant',
                'content': result['answer'],
                'sources': result['sources'],
                'timestamp': datetime.now().isoformat()
            })
            
            # Display the response
            st.markdown(result['answer'])
            
            # Display sources
            if result['sources']:
                with st.expander("📚 View Sources", expanded=True):
                    for i, source in enumerate(result['sources'][:5], 1):
                        st.markdown(f"""
                        <div class="source-card">
                            <span class="source-product">Product:</span> {source.get('product', 'Unknown')} &nbsp;|&nbsp;
                            <span class="source-issue">Issue:</span> {source.get('issue', 'Unknown')} &nbsp;|&nbsp;
                            <span class="source-score">Score:</span> {source.get('score', 0):.3f}
                            <div class="source-text">{source.get('snippet', '')}</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No sources were retrieved for this query.")

# ============================================
# MAIN APP
# ============================================

def main():
    """Main application"""
    
    # Initialize session state
    init_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Main content area
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="main-header">🔍 CrediTrust Complaint Intelligence</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Ask questions about customer complaints across financial products</div>', unsafe_allow_html=True)
    
    # Initialize pipeline
    if st.session_state.pipeline is None:
        st.session_state.pipeline = init_pipeline()
    
    if st.session_state.pipeline is None:
        st.error("❌ Failed to initialize RAG pipeline. Please check your setup.")
        st.stop()
    
    # Display existing messages
    if not st.session_state.messages:
        display_welcome()
    else:
        for message in st.session_state.messages:
            display_message(message)
    
    # Chat input
    st.markdown("---")
    
    # Get user input
    question = st.chat_input("Ask a question about customer complaints...")
    
    if question:
        # Add user message
        st.session_state.messages.append({
            'role': 'user',
            'content': question,
            'timestamp': datetime.now().isoformat()
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(question)
        
        # Process question
        process_question(
            question,
            st.session_state.pipeline,
            st.session_state.product_filter,
            st.session_state.top_k_slider
        )
        
        # Rerun to update UI
        st.rerun()

# ============================================
# ENTRY POINT
# ============================================

if __name__ == "__main__":
    main()
