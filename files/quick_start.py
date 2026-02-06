#!/usr/bin/env python3
"""
Quick Start Script for Autonomous Coding Agent
Makes it easy to get started without command line arguments
"""

import os
import sys


def get_api_key(backend):
    """Get API key from user"""
    print(f"\nEnter your {backend.upper()} API key")
    print("(or press Enter to use environment variable)")
    api_key = input("> ").strip()
    
    if not api_key:
        env_vars = {
            "gemini": "GEMINI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY"
        }
        api_key = os.environ.get(env_vars[backend], "")
        
        if not api_key:
            print(f"❌ No API key found. Set {env_vars[backend]} environment variable or enter it directly.")
            sys.exit(1)
    
    return api_key


def main():
    print("=" * 60)
    print("Autonomous Coding Agent - Quick Start")
    print("=" * 60)
    
    # Choose backend
    print("\nChoose AI Backend:")
    print("1. Google Gemini (recommended for starting)")
    print("2. Anthropic Claude")
    print("3. OpenAI GPT")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    backend_map = {
        "1": ("gemini", "gemini-1.5-flash"),
        "2": ("anthropic", "claude-sonnet-4-5-20250929"),
        "3": ("openai", "gpt-4")
    }
    
    if choice not in backend_map:
        print("❌ Invalid choice")
        sys.exit(1)
    
    backend, model = backend_map[choice]
    
    # Get API key
    api_key = get_api_key(backend)
    
    # Choose mode
    print("\nChoose Mode:")
    print("1. Interactive (talk to the agent)")
    print("2. Autonomous (agent runs continuously)")
    
    mode_choice = input("\nEnter choice (1-2): ").strip()
    mode = "interactive" if mode_choice == "1" else "auto"
    
    # Set interval for autonomous mode
    interval = 60
    if mode == "auto":
        print("\nHow often should the agent run?")
        print("1. Every 30 seconds (fast)")
        print("2. Every 1 minute (default)")
        print("3. Every 2 minutes")
        print("4. Every 5 minutes")
        print("5. Custom")
        
        interval_choice = input("\nEnter choice (1-5): ").strip()
        
        interval_map = {
            "1": 30,
            "2": 60,
            "3": 120,
            "4": 300
        }
        
        if interval_choice == "5":
            interval = int(input("Enter seconds: ").strip())
        else:
            interval = interval_map.get(interval_choice, 60)
    
    # Workspace
    print("\nWorkspace directory (default: ./workspace):")
    workspace = input("> ").strip() or "./workspace"
    
    # Enable code execution?
    print("\nEnable code execution? (y/n, default: y):")
    enable_exec = input("> ").strip().lower() != "n"
    
    # Build command
    cmd = [
        "python3",
        "autonomous_agent.py",
        "--backend", backend,
        "--api-key", api_key,
        "--model", model,
        "--workspace", workspace,
        "--mode", mode,
        "--interval", str(interval)
    ]
    
    if not enable_exec:
        cmd.append("--disable-execution")
    
    print("\n" + "=" * 60)
    print("Starting Agent...")
    print("=" * 60)
    print(f"Backend: {backend}")
    print(f"Model: {model}")
    print(f"Mode: {mode}")
    if mode == "auto":
        print(f"Interval: {interval} seconds")
    print(f"Workspace: {workspace}")
    print(f"Code Execution: {'Enabled' if enable_exec else 'Disabled'}")
    print("=" * 60 + "\n")
    
    # Run the agent
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
