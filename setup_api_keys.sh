#!/bin/bash

# Setup script for API keys
# IMPORTANT: Never commit this file with real API keys to version control!

echo "🔑 Setting up API keys..."

# Google API Key (for Gemini 2.5 Flash)
export GOOGLE_API_KEY="AIzaSyAtrs4_AhxL6E6LxuOe8JBegATeVJDdlNA"

# LlamaCloud API Key (for future LlamaCloud services)
export LLAMA_API_KEY="llx-UpcyqMBaeff02E1TKhO312bGqygqksudmmHyH4ZvXCva6E3V"

echo "✓ Google API key set"
echo "✓ LlamaCloud API key set"
echo ""
echo "⚠️  SECURITY WARNING:"
echo "   - These keys are set for this terminal session only"
echo "   - For permanent setup, add to your ~/.zshrc or ~/.bashrc:"
echo "     export GOOGLE_API_KEY=\"your-key-here\""
echo "     export LLAMA_API_KEY=\"your-key-here\""
echo "   - Or create a .env file (and add it to .gitignore)"
echo ""
echo "To make it permanent, run:"
echo "  echo 'export GOOGLE_API_KEY=\"AIzaSyAtrs4_AhxL6E6LxuOe8JBegATeVJDdlNA\"' >> ~/.zshrc"
echo "  echo 'export LLAMA_API_KEY=\"llx-UpcyqMBaeff02E1TKhO312bGqygqksudmmHyH4ZvXCva6E3V\"' >> ~/.zshrc"
echo "  source ~/.zshrc"

