"""
Interactive RAG Demo
Allows users to test the RAG pipeline with custom questions
"""

import sys
sys.path.append('notebooks')

from rag_pipeline import RAGPipeline

def interactive_demo():
    """Interactive RAG demo"""
    
    print("=" * 70)
    print("CREDITRUST FINANCIAL - RAG INTERACTIVE DEMO")
    print("=" * 70)
    print("\nAsk questions about customer complaints.")
    print("Type 'exit' to quit.")
    print("Type 'filter: ProductName' to filter by product.")
    print()
    
    # Initialize pipeline
    pipeline = RAGPipeline(top_k=5)
    
    product_filter = None
    
    while True:
        # Get user input
        question = input("\n🔍 Your question: ")
        
        if question.lower() == 'exit':
            print("Goodbye!")
            break
        
        # Check for filter
        if question.lower().startswith('filter:'):
            product_filter = question.split(':', 1)[1].strip()
            print(f"✅ Filter set to: {product_filter}")
            continue
        
        # Query pipeline
        result = pipeline.query(question, product_filter)
        
        # Display results
        print("\n" + "-" * 60)
        print("📝 Answer:")
        print(result['answer'])
        
        print("\n📚 Sources:")
        for i, source in enumerate(result['sources'][:3], 1):
            print(f"   {i}. Product: {source['product']}")
            print(f"      Issue: {source['issue']}")
            print(f"      Score: {source['score']:.4f}")
            print(f"      Snippet: {source['snippet'][:100]}...")
        print("-" * 60)

if __name__ == "__main__":
    interactive_demo()
