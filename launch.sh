#!/bin/bash
# ClaudeMood Launcher Script

echo "🎭 Starting ClaudeMood - Developer Sentiment Tracker"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Set Hugging Face cache directory
export HF_HOME="$HOME/ClaudeMood/models/hf_cache"

# Activate virtual environment and run app
cd "$HOME/ClaudeMood"

if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Using /tmp/robbert-venv"
    PYTHON="/tmp/robbert-venv/bin/python"
else
    PYTHON="$HOME/ClaudeMood/venv/bin/python"
fi

echo "🚀 Launching ClaudeMood..."
$PYTHON src/claudemood_app.py
