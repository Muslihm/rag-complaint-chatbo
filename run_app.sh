# Run the Streamlit app

echo "=========================================="
echo "CrediTrust Complaint Intelligence App"
echo "=========================================="

# Activate virtual environment
source venv/Scripts/activate 2>/dev/null || source venv/bin/activate

# Install streamlit if not installed
pip install streamlit -q

# Run the app
streamlit run app.py --server.port 8501 --server.address localhost
