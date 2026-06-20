#!/bin/bash
# Run EDA Script

echo "=========================================="
echo "Running CrediTrust EDA Pipeline"
echo "=========================================="

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.9+"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/Scripts/activate 2>/dev/null || source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Check if data exists
if [ ! -f "data/raw/complaints.csv" ]; then
    echo "⚠️ Warning: data/raw/complaints.csv not found"
    echo "   Please download the CFPB data and place it there"
    echo "   Or create a sample CSV for testing"
    
    # Create sample data for testing
    echo "📝 Creating sample data for testing..."
    mkdir -p data/raw
    cat > data/raw/complaints.csv << 'EOF'
Date received,Product,Issue,Consumer complaint narrative,Company,State
2023-01-01,Credit card,Billing dispute,I was charged incorrectly on my credit card statement. The charge was for $500 that I never made.,Test Bank,NY
2023-01-02,Credit card,Fraud,Someone opened a credit card in my name and made fraudulent purchases.,Fraud Bank,CA
2023-01-03,Personal loan,Interest rate,The interest rate on my personal loan increased without notice.,Loan Bank,TX
2023-01-04,Personal loan,Fees,I was charged excessive fees on my personal loan.,Loan Bank,FL
2023-01-05,Savings account,Fees,My savings account was charged monthly maintenance fees without disclosure.,Savings Bank,IL
2023-01-06,Savings account,Account access,I cannot access my savings account online.,Savings Bank,PA
2023-01-07,Money transfer,Delay,My international money transfer was delayed by 5 business days.,Transfer Bank,WA
2023-01-08,Money transfer,Fraud,I sent money through the transfer service and it was never received.,Transfer Bank,OH
2023-01-09,Credit card,Customer service,I called customer service 5 times about a billing issue and got different answers each time.,Test Bank,TX
2023-01-10,Personal loan,Billing dispute,My loan payment was applied incorrectly causing a late fee.,Loan Bank,CA
EOF
    echo "✅ Sample data created at data/raw/complaints.csv"
fi

# Run the EDA script
echo ""
echo "🚀 Running EDA Preprocessing..."
python notebooks/01_eda_preprocessing.py

# Check results
if [ -f "data/filtered_complaints.csv" ]; then
    echo ""
    echo "✅ Pipeline completed successfully!"
    echo "   - Filtered data: data/filtered_complaints.csv"
    echo "   - EDA Report: data/processed/eda_report.txt"
    echo "   - Visualizations: data/processed/eda_visualizations.png"
else
    echo "❌ Pipeline failed. Check error messages above."
fi
