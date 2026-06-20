
"""
Test script for vector store functionality
"""

import os
import sys
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

def test_vector_store():
    """Test the vector store loading and searching"""
    
    print("=" * 70)
    print("TESTING VECTOR STORE")
    print("=" * 70)
    
    # Load vector store
    try:
        import faiss
        
        # Load FAISS index
        index_path = 'vector_store/complaint_index.faiss'
        index = faiss.read_index(index_path)
        print(f"✅ FAISS index loaded: {index.ntotal} vectors")
        
        # Load metadata
        metadata_path = 'vector_store/complaint_index_metadata.pkl'
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            metadata = data['metadata']
        print(f"✅ Metadata loaded: {len(metadata)} entries")
        
        # Load embedding model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded")
        
        # Test queries
        test_queries = [
            "What are the most common credit card complaints?",
            "Why are customers unhappy with money transfers?",
            "Tell me about fraud complaints",
            "What issues do customers have with fees?",
            "Compare credit card and personal loan complaints"
        ]
        
        print("\n🔍 Testing Search Queries:")
        print("-" * 70)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Query: '{query}'")
            
            # Generate embedding
            query_embedding = model.encode([query])[0]
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            
            # Search
            distances, indices = index.search(query_embedding, 3)
            
            print("   Top Results:")
            for j, idx in enumerate(indices[0]):
                if idx < len(metadata):
                    meta = metadata[idx]
                    score = distances[0][j]
                    print(f"   - [{j+1}] Product: {meta['product_category']:20s} Score: {score:.4f}")
                    print(f"       Issue: {meta['issue'][:40]}...")
        
        print("\n" + "=" * 70)
        print("✅ Vector store test completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vector_store()
