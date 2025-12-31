#!/usr/bin/env python3
"""
Setup script to create .env file from .env.example
"""

import os
from pathlib import Path

def setup_env():
    """Create .env file if it doesn't exist."""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print("✓ .env file already exists")
        return
    
    if env_example.exists():
        # Copy .env.example to .env
        with open(env_example, 'r') as f:
            content = f.read()
        
        with open(env_file, 'w') as f:
            f.write(content)
        
        print("✓ Created .env file from .env.example")
        print("⚠️  Please edit .env and add your actual API keys")
    else:
        # Create a basic .env file
        env_content = """# API Keys for LlamaIndex Document Chatbot
# This file is automatically loaded by the application
# Never commit this file to version control!

# Google API Key (Required for Gemini 2.5 Flash)
# Get it from: https://aistudio.google.com/
GOOGLE_API_KEY=your-google-api-key-here

# LlamaCloud API Key (Optional - for future LlamaCloud services)
# Get it from: https://cloud.llamaindex.ai/
# Note: Currently using free Hugging Face embeddings (no API key needed)
LLAMA_API_KEY=your-llama-api-key-here
"""
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        print("✓ Created .env file")
        print("⚠️  Please edit .env and add your actual API keys")

if __name__ == "__main__":
    setup_env()

