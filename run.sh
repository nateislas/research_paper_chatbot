#!/bin/bash

# Quick start script for LlamaIndex Document Chatbot

echo "🚀 Starting LlamaIndex Document Chatbot..."

# Check if we're already in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    # Not in a venv, check if local venv exists
    if [ -d "venv" ]; then
        echo "📦 Activating local virtual environment..."
        source venv/bin/activate
    else
        echo "⚠️  Warning: No virtual environment detected"
        echo "   You can either:"
        echo "   1. Create one: python3.10 -m venv venv && source venv/bin/activate"
        echo "   2. Continue with system Python (not recommended)"
        read -p "   Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
else
    echo "✓ Virtual environment detected: $VIRTUAL_ENV"
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found"
    echo "   Creating .env from .env.example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "   ✓ Created .env file - please add your API keys"
    else
        echo "   Please create a .env file with your API keys:"
        echo "   GOOGLE_API_KEY=your-google-api-key"
        echo "   LLAMA_API_KEY=your-llama-api-key"
    fi
fi

# Check if API keys are set (will be loaded from .env by python-dotenv)
echo "✓ Environment variables will be loaded from .env file"

# Check if streamlit is installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "📦 Installing Streamlit..."
    pip install streamlit
fi

# Start Streamlit app
echo "🌐 Starting Streamlit web interface..."
echo "   Open your browser to: http://localhost:8501"
echo ""
streamlit run research_paper_chatbot/app.py

