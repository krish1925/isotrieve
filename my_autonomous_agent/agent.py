import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def configure_agent():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "INSERT_YOUR_GEMINI_KEY_HERE":
        print("Error: GEMINI_API_KEY not found in .env file or is still the placeholder.")
        print("Please set your Gemini API key in my_autonomous_agent/.env")
        return None
    
    genai.configure(api_key=api_key)
    
    # Use a safe default model
    model = genai.GenerativeModel('gemini-pro')
    return model

def chat_loop(model):
    chat = model.start_chat(history=[])
    print("Agent: Hello! I am your autonomous agent. How can I help you today? (Type 'exit' to quit)")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Agent: Goodbye!")
            break
            
        try:
            response = chat.send_message(user_input)
            print(f"Agent: {response.text}")
        except Exception as e:
            print(f"Agent: An error occurred: {e}")

def main():
    print("Initializing Autonomous Agent...")
    model = configure_agent()
    if model:
        print("Agent initialized successfully using gemini-pro.")
        chat_loop(model)

if __name__ == "__main__":
    main()
