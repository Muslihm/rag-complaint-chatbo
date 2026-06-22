"""
RAG Pipeline Evaluation Script
Runs comprehensive evaluation and generates detailed report
"""

import os
import sys
import json
from datetime import datetime

# Add notebooks directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'notebooks'))

# Import from the renamed module
from rag_pipeline import RAGPipeline, evaluate_rag_pipeline, generate_evaluation_report

def run_evaluation():
    """Run complete evaluation"""
    
    print("=" * 70)
    print("RAG PIPELINE EVALUATION")
    print("=" * 70)
    
    # Check if required files exist
    required_files = [
        'vector_store/complaint_index_light.faiss',
        'vector_store/complaint_index_light_metadata.pkl',
        'data/filtered_complaints.csv'
    ]
    
    missing = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing.append(file_path)
    
    if missing:
        print("\n[WARNING] Missing required files:")
        for file_path in missing:
            print(f"   - {file_path}")
        print("\nPlease run Task 1 and Task 2 first.")
        return None
    
    # Initialize pipeline
    print("\n[OK] All required files found.")
    pipeline = RAGPipeline(top_k=5)
    
    # Run evaluation
    results = evaluate_rag_pipeline(pipeline)
    
    # Generate report
    report = generate_evaluation_report(results)
    
    # Save report
    os.makedirs('data/processed', exist_ok=True)
    report_path = 'data/processed/rag_evaluation_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n[OK] Report saved to: {report_path}")
    
    # Save results as JSON
    json_path = 'data/processed/rag_evaluation_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_questions': len(results),
            'results': results
        }, f, indent=2, default=str)
    
    print(f"[OK] Results saved to: {json_path}")
    
    print("\n" + "=" * 70)
    print("[OK] EVALUATION COMPLETE")
    print("=" * 70)
    
    return results

if __name__ == "__main__":
    run_evaluation()