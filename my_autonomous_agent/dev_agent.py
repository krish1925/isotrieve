import os
import sys
import time
import subprocess
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY or API_KEY == "INSERT_YOUR_GEMINI_KEY_HERE":
    print(Fore.RED + "Error: GEMINI_API_KEY not found or invalid in .env")
    sys.exit(1)

client = genai.Client(api_key=API_KEY)

# --- Tool Definitions ---

def execute_command(command: str):
    """Executes a shell command and returns the output."""
    print(Fore.CYAN + f"Executing: {command}")
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, cwd=os.getcwd()
        )
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        return output[:5000] # Truncate massive outputs
    except Exception as e:
        return f"Error executing command: {str(e)}"

def read_file(path: str):
    """Reads the content of a file."""
    print(Fore.CYAN + f"Reading: {path}")
    try:
        with open(path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def write_file(path: str, content: str):
    """Writes content to a file."""
    print(Fore.CYAN + f"Writing to: {path}")
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

def list_files(path: str = "."):
    """Lists files in a directory."""
    print(Fore.CYAN + f"Listing: {path}")
    try:
        return str(os.listdir(path))
    except Exception as e:
        return f"Error listing directory: {str(e)}"

# Define tools for the new SDK
tools = [execute_command, read_file, write_file, list_files]

# --- Global Stats & Rate Limiting ---
REQUEST_COUNT = 0
LAST_REQUEST_TIME = 0
MAX_RPM = 5
SECONDS_BETWEEN_REQUESTS = 60 / MAX_RPM  # 12 seconds to be safe
STATE_FILE = "my_autonomous_agent/state.json"
USER_INSTRUCTIONS_FILE = "my_autonomous_agent/user_instructions.md"

def save_state(context_summary, session_id):
    state = {
        "request_count": REQUEST_COUNT,
        "context_summary": context_summary,
        "session_id": session_id
    }
    try:
        # Sanitize summary for JSON safety (ensure no raw control characters)
        # However json.dump handles this well if summary is a string.
        # The issue is usually manual edits breaking the JSON.
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
        # Verbose only on session ends to avoid clutter, but we save every request.
    except Exception as e:
        print(Fore.RED + f"Error saving state: {e}")

def load_state():
    global REQUEST_COUNT
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                content = f.read()
                # Use strict=False to handle potential manual edit errors (like raw newlines)
                state = json.loads(content, strict=False)
            REQUEST_COUNT = state.get("request_count", 0)
            print(Fore.BLUE + f"State loaded from {STATE_FILE}. Request Count: {REQUEST_COUNT}")
            return state.get("context_summary", "Starting fresh."), state.get("session_id", 1)
        except Exception as e:
            print(Fore.RED + f"Error loading state (likely malformed JSON): {e}")
            print(Fore.YELLOW + "Attempting to continue with fresh state but keeping Request Count if possible.")
    return "Starting fresh. No previous actions taken.", 1

def load_user_instructions():
    if os.path.exists(USER_INSTRUCTIONS_FILE):
        try:
            with open(USER_INSTRUCTIONS_FILE, "r") as f:
                return f.read().strip()
        except:
            pass
    return ""

def rate_limited_send_message(chat, prompt, context_summary, session_id):
    global REQUEST_COUNT, LAST_REQUEST_TIME
    
    current_time = time.time()
    elapsed = current_time - LAST_REQUEST_TIME
    
    if elapsed < SECONDS_BETWEEN_REQUESTS:
        wait_time = SECONDS_BETWEEN_REQUESTS - elapsed
        print(Fore.YELLOW + f"Rate limiting: Waiting {wait_time:.2f}s...")
        time.sleep(wait_time)
    
    REQUEST_COUNT += 1
    LAST_REQUEST_TIME = time.time()
    print(Fore.WHITE + f"[Request #{REQUEST_COUNT}] Sending message...")
    
    # NEW: Save state BEFORE sending to ensure Request Count is persisted immediately
    save_state(context_summary, session_id)
    
    return chat.send_message(prompt)

# --- Agent Logic ---

def load_instructions():
    try:
        with open("my_autonomous_agent/instructions.md", "r") as f:
            return f.read()
    except:
        return "You are a helpful developer agent."

def run_session(context_summary, session_id):
    """Runs a single 'session' of the agent conversation."""
    print(Fore.GREEN + f"\n--- Starting Session {session_id} ---")
    
    system_instruction = load_instructions()
    
    # Load User Instructions
    user_instr = load_user_instructions()
    if user_instr:
        system_instruction += f"\n\nCURRENT USER GUIDANCE/OVERRIDE:\n{user_instr}\n\nStrictly follow these user instructions above all else."

    if context_summary:
        system_instruction += f"\n\nPREVIOUS SESSION CONTEXT:\n{context_summary}\n\nContinue from here."

    model_id = 'gemini-3-flash-preview'
    
    # Initialize chat session
    chat = client.chats.create(
        model=model_id,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=tools,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=False
            )
        )
    )

    # Initial prompt to kick off or continue work
    prompt = "Proceed with your task. If you are just starting, explore the environment. If continuing, look at the Context Summary."
    
    turns = 0
    max_turns = 10 
    
    try:
        response = rate_limited_send_message(chat, prompt, context_summary, session_id)
        print(Fore.MAGENTA + f"Agent: {response.text}")
        
        while turns < max_turns:
            continuation_prompt = "Continue. If you have completed a significant amount of work or if ~1 minute has passed, please output 'SUMMARY: ' followed by a concise summary of what you did and what needs to be done next, then stop."
            
            response = rate_limited_send_message(chat, continuation_prompt, context_summary, session_id)
            text = response.text
            print(Fore.MAGENTA + f"Agent: {text}")
            
            # Update local context summary with whatever the agent said as a interim progress
            if text:
                context_summary = f"Last turn response: {text[:500]}..." 
            
            if text and "SUMMARY:" in text:
                summary = text.split("SUMMARY:", 1)[1].strip()
                save_state(summary, session_id + 1)
                return summary
            
            turns += 1

    except Exception as e:
        print(Fore.RED + f"Session Error: {e}")
        save_state(f"Session crashed. Last context: {context_summary[:1000]}", session_id)
        return f"Session crashed. Last known state: {context_summary}"

    return "Session ended without explicit summary. Check logs."

def main():
    print(Fore.BLUE + "Initializing Autonomous Developer Agent (v2.3 with Robust Persistence)...")
    
    context_summary, session_id = load_state()
    
    while True:
        try:
            new_summary = run_session(context_summary, session_id)
            
            print(Fore.BLUE + f"\n--- Session {session_id} Finished ---")
            print(Fore.WHITE + f"New Context: {new_summary}")
            print(Fore.CYAN + f"Total Requests Made So Far: {REQUEST_COUNT}")
            
            context_summary = new_summary
            session_id += 1
            
            # Auto-save at end of session
            save_state(context_summary, session_id)
            
            if session_id > 20: 
                print(Fore.RED + "Stopping after 20 sessions for safety.")
                break
                
            print("Restarting in 5 seconds...")
            time.sleep(5)
            
        except KeyboardInterrupt:
            print(f"\nUser stopped the agent. Total requests: {REQUEST_COUNT}")
            save_state(context_summary, session_id)
            break

if __name__ == "__main__":
    main()
