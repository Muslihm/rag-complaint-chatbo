@echo off
echo ==========================================
echo CrediTrust Complaint Intelligence App
echo ==========================================

call venv\Scripts\activate.bat
pip install streamlit -q
streamlit run app.py --server.port 8501 --server.address localhost

