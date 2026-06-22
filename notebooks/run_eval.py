import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_pipeline import RAGPipeline, evaluate_rag_pipeline, generate_evaluation_report

print("=" * 70)
print("RAG PIPELINE EVALUATION")
print("=" * 70)

# Check required files
required = ['vector_store/complaint_index_light.faiss', 'data/filtered_complaints.csv']
missing = [f for f in required if not os.path.exists(f)]
if missing:
    print("\n[WARNING] Missing files:", missing)

# Run evaluation
pipeline = RAGPipeline(top_k=5)
results = evaluate_rag_pipeline(pipeline)
report = generate_evaluation_report(results)

# Save report
os.makedirs('data/processed', exist_ok=True)
report_path = 'data/processed/rag_evaluation_report.md'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\n[OK] Report saved to: {report_path}")
print("=" * 70)
