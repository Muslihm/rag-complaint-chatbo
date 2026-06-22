"""
Test the Streamlit App Functionality
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'notebooks'))

try:
    from rag_pipeline import RAGPipeline
    
    print("=" * 60)
    print("Testing RAG Pipeline for Streamlit App")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = RAGPipeline(top_k=5)
    print("[OK] Pipeline initialized")
    
    # Test query
    test_questions = [
        "What are common credit card complaints?",
        "Tell me about fraud issues",
        "Compare credit cards and personal loans"
    ]
    
    for q in test_questions:
        print(f"\n[TEST] {q}")
        result = pipeline.query(q)
        print(f"  Answer: {result['answer'][:100]}...")
        print(f"  Sources: {result['num_sources']}")
    
    print("\n[OK] All tests passed!")
    print("\nYou can now run the Streamlit app:")
    print("  streamlit run app.py")
    
except Exception as e:
    print(f"[ERROR] {e}")
