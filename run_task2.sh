
#!/bin/bash
# Task 2 Launcher

echo "=========================================="
echo "Running Task 2: Chunking & Embedding"
echo "=========================================="

# Set memory optimization
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Activate virtual environment
source venv/Scripts/activate 2>/dev/null || source venv/bin/activate

# Install required packages
echo "📦 Installing Task 2 dependencies..."
pip install sentence-transformers faiss-cpu tqdm

# Run Task 2
python notebooks/02_chunking_embedding_indexing.py

echo ""
echo "✅ Task 2 completed!"


