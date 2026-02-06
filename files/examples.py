#!/usr/bin/env python3
"""
Example script showing how to use the Autonomous Agent programmatically
"""

from autonomous_agent import AutonomousAgent, AgentConfig


def example_basic_usage():
    """Basic usage example"""
    config = AgentConfig(
        ai_backend="gemini",
        api_key="YOUR_GEMINI_API_KEY",
        model="gemini-1.5-flash",
        run_interval_seconds=60,
        workspace_dir="./my_project"
    )
    
    agent = AutonomousAgent(config)
    
    # Run a single iteration with a task
    agent.run_iteration(
        "Create a Python script that calculates fibonacci numbers"
    )


def example_autonomous_coding():
    """Example of autonomous coding session"""
    config = AgentConfig(
        ai_backend="gemini",
        api_key="YOUR_API_KEY",
        run_interval_seconds=120,  # Run every 2 minutes
        workspace_dir="./coding_workspace",
        enable_code_execution=True,
        safe_execution_only=True
    )
    
    agent = AutonomousAgent(config)
    
    # Give initial task
    agent.run_iteration(
        """I need you to create a simple REST API with the following features:
        1. A GET endpoint that returns a list of users
        2. A POST endpoint that adds a new user
        3. Use Flask and store data in memory
        4. Include proper error handling
        
        Work on this step by step, and test your code as you go."""
    )
    
    # Let it run autonomously
    agent.run_loop()


def example_with_custom_tools():
    """Example with custom system prompt and tools"""
    config = AgentConfig(
        ai_backend="anthropic",
        api_key="YOUR_ANTHROPIC_API_KEY",
        model="claude-sonnet-4-5-20250929",
        run_interval_seconds=90,
        workspace_dir="./custom_workspace",
        system_prompt_file="./custom_system_prompt.txt"
    )
    
    agent = AutonomousAgent(config)
    
    # Customize the system prompt
    custom_prompt = """You are a specialized agent focused on creating well-tested code.

Your workflow:
1. Understand the requirements
2. Create the main code file
3. Create comprehensive tests
4. Run the tests
5. Fix any issues
6. Document the code

Always follow test-driven development principles.

Available tools: read_file, write_file, edit_file, list_files, search_files, 
execute_code, run_command, read_system_prompt, edit_system_prompt

Use the same <tool_call> format as before.
"""
    
    agent._tool_edit_system_prompt(custom_prompt)
    
    # Start working
    agent.run_iteration("Create a calculator module with unit tests")


def example_file_refactoring():
    """Example of using the agent for code refactoring"""
    config = AgentConfig(
        ai_backend="gemini",
        api_key="YOUR_API_KEY",
        workspace_dir="./refactor_workspace"
    )
    
    agent = AutonomousAgent(config)
    
    # First, let's say you have a messy file
    agent.run_iteration(
        """There's a file called messy_code.py in the workspace. 
        Please refactor it to:
        - Follow PEP 8 style guidelines
        - Add proper docstrings
        - Extract repeated code into functions
        - Add type hints
        - Improve variable names"""
    )


def example_fast_iteration():
    """Example with fast iteration for quick tasks"""
    config = AgentConfig(
        ai_backend="gemini",
        api_key="YOUR_API_KEY",
        run_interval_seconds=30,  # Run every 30 seconds
        workspace_dir="./quick_tasks"
    )
    
    agent = AutonomousAgent(config)
    
    agent.run_iteration(
        """Create 5 different sorting algorithm implementations:
        1. Bubble sort
        2. Quick sort
        3. Merge sort
        4. Heap sort
        5. Insertion sort
        
        Each should be in its own file with examples."""
    )
    
    agent.run_loop()


def example_disable_execution():
    """Example with code execution disabled (safer)"""
    config = AgentConfig(
        ai_backend="gemini",
        api_key="YOUR_API_KEY",
        enable_code_execution=False,  # Only file operations
        workspace_dir="./safe_workspace"
    )
    
    agent = AutonomousAgent(config)
    
    agent.run_iteration(
        "Create a comprehensive documentation file for a REST API project"
    )


def example_interactive_session():
    """Example of interactive mode"""
    config = AgentConfig(
        ai_backend="gemini",
        api_key="YOUR_API_KEY",
        workspace_dir="./interactive_workspace"
    )
    
    agent = AutonomousAgent(config)
    
    # Run in interactive mode
    agent.run_interactive()


def example_multi_step_project():
    """Example of a multi-step project"""
    config = AgentConfig(
        ai_backend="anthropic",
        api_key="YOUR_API_KEY",
        run_interval_seconds=180,  # 3 minutes between steps
        workspace_dir="./web_app"
    )
    
    agent = AutonomousAgent(config)
    
    # Step 1: Project setup
    agent.run_iteration(
        """Let's build a simple blog application. First step:
        Create the project structure with:
        - app.py (main Flask app)
        - models.py (data models)
        - templates/ directory
        - static/ directory
        - requirements.txt
        
        Just create the files with basic structure."""
    )
    
    # The agent will continue autonomously
    agent.run_loop()


def example_with_constraints():
    """Example with specific constraints"""
    config = AgentConfig(
        ai_backend="gemini",
        api_key="YOUR_API_KEY",
        workspace_dir="./constrained_project",
        allowed_file_extensions=[".py", ".txt", ".md"],  # Only these types
        max_file_size_kb=100,  # Smaller files only
        safe_execution_only=True
    )
    
    agent = AutonomousAgent(config)
    
    agent.run_iteration(
        """Create a command-line tool that:
        - Takes a text file as input
        - Counts word frequency
        - Outputs results to another text file
        - All code must be in a single Python file under 100KB"""
    )


def example_research_and_implement():
    """Example where agent researches then implements"""
    config = AgentConfig(
        ai_backend="anthropic",
        api_key="YOUR_API_KEY",
        run_interval_seconds=120,
        workspace_dir="./research_project"
    )
    
    agent = AutonomousAgent(config)
    
    agent.run_iteration(
        """Research and implement the A* pathfinding algorithm.
        
        Steps:
        1. First, read about A* and understand it
        2. Create a simple grid-based implementation
        3. Add visualization (ASCII art is fine)
        4. Create test cases
        5. Document how it works
        
        Take your time and do this thoroughly."""
    )
    
    agent.run_loop()


# Run examples
if __name__ == "__main__":
    print("Autonomous Agent Examples")
    print("=" * 60)
    print("\nUncomment the example you want to run\n")
    
    # Uncomment one of these to run:
    
    # example_basic_usage()
    # example_autonomous_coding()
    # example_with_custom_tools()
    # example_file_refactoring()
    # example_fast_iteration()
    # example_disable_execution()
    # example_interactive_session()
    # example_multi_step_project()
    # example_with_constraints()
    # example_research_and_implement()
    
    print("\nChoose an example and uncomment it in the code!")
