#!/bin/bash

# Move to script location (your project root)
cd "$(dirname "$0")"

echo "🔧 Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+ and try again."
    exit 1
fi

echo "📦 Creating virtual environment..."
python3 -m venv venv

echo "📦 Activating virtual environment..."
source venv/bin/activate

echo "⬇️ Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "🚀 Launching DeepDisco..."
bash run_app.sh
