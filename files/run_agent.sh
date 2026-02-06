#!/bin/bash
# Quick launcher for Autonomous Coding Agent

echo "=================================="
echo "Autonomous Coding Agent Launcher"
echo "=================================="
echo

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not found"
    echo "Please install Python 3 and try again"
    exit 1
fi

# Check if dependencies are installed
if ! python3 -c "import google.generativeai" &> /dev/null && \
   ! python3 -c "import anthropic" &> /dev/null && \
   ! python3 -c "import openai" &> /dev/null; then
    echo "⚠️  Dependencies not found. Installing..."
    pip3 install -r requirements.txt
    echo
fi

# Check for API keys in environment
GEMINI_KEY="${GEMINI_API_KEY:-}"
ANTHROPIC_KEY="${ANTHROPIC_API_KEY:-}"
OPENAI_KEY="${OPENAI_API_KEY:-}"

echo "Select launcher mode:"
echo "1. Quick Start (guided setup)"
echo "2. Basic Agent (standard features)"
echo "3. Enhanced Agent (with logging and stats)"
echo "4. Custom command"
echo

read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo
        echo "Starting Quick Start..."
        python3 quick_start.py
        ;;
    2)
        echo
        echo "Select AI Backend:"
        echo "1. Google Gemini"
        echo "2. Anthropic Claude"
        echo "3. OpenAI GPT"
        echo
        read -p "Enter choice (1-3): " backend_choice
        
        case $backend_choice in
            1) 
                BACKEND="gemini"
                API_KEY="${GEMINI_KEY}"
                ;;
            2) 
                BACKEND="anthropic"
                API_KEY="${ANTHROPIC_KEY}"
                ;;
            3) 
                BACKEND="openai"
                API_KEY="${OPENAI_KEY}"
                ;;
            *)
                echo "❌ Invalid choice"
                exit 1
                ;;
        esac
        
        if [ -z "$API_KEY" ]; then
            read -p "Enter your ${BACKEND^^} API key: " API_KEY
        fi
        
        echo
        echo "Starting Basic Agent..."
        python3 autonomous_agent.py --backend "$BACKEND" --api-key "$API_KEY"
        ;;
    3)
        echo
        echo "Select AI Backend:"
        echo "1. Google Gemini"
        echo "2. Anthropic Claude"
        echo "3. OpenAI GPT"
        echo
        read -p "Enter choice (1-3): " backend_choice
        
        case $backend_choice in
            1) 
                BACKEND="gemini"
                API_KEY="${GEMINI_KEY}"
                ;;
            2) 
                BACKEND="anthropic"
                API_KEY="${ANTHROPIC_KEY}"
                ;;
            3) 
                BACKEND="openai"
                API_KEY="${OPENAI_KEY}"
                ;;
            *)
                echo "❌ Invalid choice"
                exit 1
                ;;
        esac
        
        if [ -z "$API_KEY" ]; then
            read -p "Enter your ${BACKEND^^} API key: " API_KEY
        fi
        
        echo
        echo "Starting Enhanced Agent with logging..."
        python3 enhanced_agent.py --backend "$BACKEND" --api-key "$API_KEY"
        ;;
    4)
        echo
        echo "Custom Command Mode"
        echo "Example: python3 autonomous_agent.py --backend gemini --api-key YOUR_KEY --interval 120 --mode auto"
        echo
        read -p "Enter your custom command: " custom_cmd
        eval "$custom_cmd"
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac
