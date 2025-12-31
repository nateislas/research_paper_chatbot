#!/bin/bash

# Setup script for API keys
# IMPORTANT: Never commit this file with real API keys to version control!
# Add your actual API keys to this file locally, but DO NOT commit them.

echo "Setting up API keys..."

# Google API Key (for Gemini 2.5 Flash)
# Replace YOUR_GOOGLE_API_KEY_HERE with your actual key
if [ -z "$GOOGLE_API_KEY" ]; then
    export GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
fi

# LlamaCloud API Key
# Replace YOUR_LLAMA_API_KEY_HERE with your actual key
if [ -z "$LLAMA_API_KEY" ]; then
    export LLAMA_API_KEY="YOUR_LLAMA_API_KEY_HERE"
fi

echo "✓ API keys loaded (check that they are not placeholders!)"
echo ""
echo "SECURITY WARNING:"
echo "   - These keys are set for this terminal session only"
echo "   - For permanent setup, add to your ~/.zshrc or ~/.bashrc:"
echo "     export GOOGLE_API_KEY=\"your-key-here\""
echo "     export LLAMA_API_KEY=\"your-key-here\""
echo "   - Or create a .env file (and add it to .gitignore)"
echo ""
echo "RECOMMENDED: Use a .env file instead:"
echo "  1. Create a .env file in the project root"
echo "  2. Add: GOOGLE_API_KEY=your-key-here"
echo "  3. Add: LLAMA_API_KEY=your-key-here"
echo "  4. The .env file is already in .gitignore"

