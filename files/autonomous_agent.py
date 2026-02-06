#!/usr/bin/env python3
"""
Autonomous Coding Agent - Custom framework for AI-powered code editing and execution
Supports multiple AI backends with configurable tools and run frequency
"""

import os
import time
import json
import subprocess
import re
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
from abc import ABC, abstractmethod


@dataclass
class AgentConfig:
    """Configuration for the autonomous agent"""
    ai_backend: str = "gemini"  # "gemini", "anthropic", "openai"
    api_key: str = ""
    model: str = ""  # e.g., "gemini-1.5-flash", "claude-sonnet-4-5", "gpt-4"
    run_interval_seconds: int = 60  # How often the agent runs automatically
    workspace_dir: str = "./workspace"
    system_prompt_file: str = "./system_prompt.txt"
    conversation_history_file: str = "./conversation_history.json"
    max_conversation_history: int = 20
    allowed_file_extensions: List[str] = field(default_factory=lambda: [
        ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".md", ".txt", ".html", ".css"
    ])
    blocked_paths: List[str] = field(default_factory=lambda: [
        "/etc", "/sys", "/proc", "/.ssh", "/home"
    ])
    max_file_size_kb: int = 500
    enable_code_execution: bool = True
    safe_execution_only: bool = True  # Only allow whitelisted commands


class AIBackend(ABC):
    """Abstract base class for AI backends"""
    
    @abstractmethod
    def generate_response(self, messages: List[Dict[str, str]], system_prompt: str) -> str:
        """Generate a response from the AI model"""
        pass


class GeminiBackend(AIBackend):
    """Google Gemini AI backend"""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        
    def generate_response(self, messages: List[Dict[str, str]], system_prompt: str) -> str:
        # Combine system prompt with conversation
        full_prompt = f"{system_prompt}\n\n"
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            full_prompt += f"{role}: {msg['content']}\n\n"
        
        response = self.model.generate_content(full_prompt)
        return response.text


class AnthropicBackend(AIBackend):
    """Anthropic Claude AI backend"""
    
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5-20250929"):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key)
        self.model = model
        
    def generate_response(self, messages: List[Dict[str, str]], system_prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            system=system_prompt,
            messages=messages
        )
        return response.content[0].text


class OpenAIBackend(AIBackend):
    """OpenAI GPT AI backend"""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        import openai
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        
    def generate_response(self, messages: List[Dict[str, str]], system_prompt: str) -> str:
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        response = self.client.chat.completions.create(
            model=self.model,
            messages=full_messages
        )
        return response.choices[0].message.content


class Tool:
    """Base class for agent tools"""
    
    def __init__(self, name: str, description: str, function: Callable):
        self.name = name
        self.description = description
        self.function = function
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool and return results"""
        try:
            result = self.function(**kwargs)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}


class AutonomousAgent:
    """Main autonomous coding agent"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.backend = self._initialize_backend()
        self.tools = self._register_tools()
        self.conversation_history = self._load_conversation_history()
        self.system_prompt = self._load_system_prompt()
        self.workspace_dir = Path(config.workspace_dir)
        self.workspace_dir.mkdir(exist_ok=True)
        
    def _initialize_backend(self) -> AIBackend:
        """Initialize the AI backend based on config"""
        backend_map = {
            "gemini": GeminiBackend,
            "anthropic": AnthropicBackend,
            "openai": OpenAIBackend
        }
        
        if self.config.ai_backend not in backend_map:
            raise ValueError(f"Unknown AI backend: {self.config.ai_backend}")
        
        backend_class = backend_map[self.config.ai_backend]
        return backend_class(self.config.api_key, self.config.model)
    
    def _load_system_prompt(self) -> str:
        """Load system prompt from file"""
        prompt_file = Path(self.config.system_prompt_file)
        if prompt_file.exists():
            return prompt_file.read_text()
        else:
            default_prompt = self._get_default_system_prompt()
            prompt_file.write_text(default_prompt)
            return default_prompt
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for the agent"""
        return """You are an autonomous coding agent with access to file editing and code execution tools.

Your capabilities:
- Read and edit files in the workspace
- Execute code and shell commands (with restrictions)
- List and search files
- Create new files and directories

When responding, use the following format to call tools:

<tool_call>
<tool_name>tool_name_here</tool_name>
<parameters>
{
  "param1": "value1",
  "param2": "value2"
}
</parameters>
</tool_call>

Available tools:
- read_file: Read contents of a file (params: filepath)
- write_file: Write content to a file (params: filepath, content)
- edit_file: Edit specific lines in a file (params: filepath, start_line, end_line, new_content)
- list_files: List files in a directory (params: directory)
- search_files: Search for files by pattern (params: pattern, directory)
- execute_code: Execute Python code (params: code)
- run_command: Run shell command (params: command)
- read_system_prompt: Read the current system prompt
- edit_system_prompt: Edit the system prompt (params: new_prompt)

Always think step-by-step and explain your reasoning before using tools.
Keep track of your goals and make progress incrementally.
"""
    
    def _load_conversation_history(self) -> List[Dict[str, str]]:
        """Load conversation history from file"""
        history_file = Path(self.config.conversation_history_file)
        if history_file.exists():
            with open(history_file, 'r') as f:
                return json.load(f)
        return []
    
    def _save_conversation_history(self):
        """Save conversation history to file"""
        # Keep only recent messages
        if len(self.conversation_history) > self.config.max_conversation_history:
            self.conversation_history = self.conversation_history[-self.config.max_conversation_history:]
        
        history_file = Path(self.config.conversation_history_file)
        with open(history_file, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)
    
    def _register_tools(self) -> Dict[str, Tool]:
        """Register all available tools"""
        tools = {}
        
        # File operations
        tools["read_file"] = Tool(
            "read_file",
            "Read contents of a file",
            self._tool_read_file
        )
        
        tools["write_file"] = Tool(
            "write_file",
            "Write content to a file",
            self._tool_write_file
        )
        
        tools["edit_file"] = Tool(
            "edit_file",
            "Edit specific lines in a file",
            self._tool_edit_file
        )
        
        tools["list_files"] = Tool(
            "list_files",
            "List files in a directory",
            self._tool_list_files
        )
        
        tools["search_files"] = Tool(
            "search_files",
            "Search for files by pattern",
            self._tool_search_files
        )
        
        # Code execution
        if self.config.enable_code_execution:
            tools["execute_code"] = Tool(
                "execute_code",
                "Execute Python code",
                self._tool_execute_code
            )
            
            tools["run_command"] = Tool(
                "run_command",
                "Run shell command",
                self._tool_run_command
            )
        
        # System prompt management
        tools["read_system_prompt"] = Tool(
            "read_system_prompt",
            "Read the current system prompt",
            self._tool_read_system_prompt
        )
        
        tools["edit_system_prompt"] = Tool(
            "edit_system_prompt",
            "Edit the system prompt",
            self._tool_edit_system_prompt
        )
        
        return tools
    
    def _is_safe_path(self, filepath: str) -> bool:
        """Check if a file path is safe to access"""
        path = Path(filepath).resolve()
        
        # Check if path is within workspace
        try:
            path.relative_to(self.workspace_dir.resolve())
        except ValueError:
            # Also allow system prompt file
            if path == Path(self.config.system_prompt_file).resolve():
                return True
            return False
        
        # Check blocked paths
        for blocked in self.config.blocked_paths:
            if str(path).startswith(blocked):
                return False
        
        return True
    
    def _is_allowed_extension(self, filepath: str) -> bool:
        """Check if file extension is allowed"""
        ext = Path(filepath).suffix
        return ext in self.config.allowed_file_extensions or ext == ""
    
    # Tool implementations
    
    def _tool_read_file(self, filepath: str) -> str:
        """Read a file's contents"""
        if not self._is_safe_path(filepath):
            raise PermissionError(f"Access denied to path: {filepath}")
        
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Check file size
        if path.stat().st_size > self.config.max_file_size_kb * 1024:
            raise ValueError(f"File too large (max {self.config.max_file_size_kb}KB)")
        
        return path.read_text()
    
    def _tool_write_file(self, filepath: str, content: str) -> str:
        """Write content to a file"""
        if not self._is_safe_path(filepath):
            raise PermissionError(f"Access denied to path: {filepath}")
        
        if not self._is_allowed_extension(filepath):
            raise PermissionError(f"File extension not allowed: {Path(filepath).suffix}")
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        
        return f"Successfully wrote {len(content)} characters to {filepath}"
    
    def _tool_edit_file(self, filepath: str, start_line: int, end_line: int, new_content: str) -> str:
        """Edit specific lines in a file"""
        if not self._is_safe_path(filepath):
            raise PermissionError(f"Access denied to path: {filepath}")
        
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        lines = path.read_text().splitlines()
        
        # Convert to 0-indexed
        start_idx = start_line - 1
        end_idx = end_line
        
        if start_idx < 0 or end_idx > len(lines):
            raise ValueError(f"Line range out of bounds: {start_line}-{end_line}")
        
        # Replace lines
        new_lines = new_content.splitlines()
        lines[start_idx:end_idx] = new_lines
        
        path.write_text('\n'.join(lines))
        
        return f"Successfully edited lines {start_line}-{end_line} in {filepath}"
    
    def _tool_list_files(self, directory: str = ".") -> str:
        """List files in a directory"""
        dir_path = self.workspace_dir / directory
        
        if not self._is_safe_path(str(dir_path)):
            raise PermissionError(f"Access denied to directory: {directory}")
        
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        files = []
        for item in sorted(dir_path.iterdir()):
            rel_path = item.relative_to(self.workspace_dir)
            if item.is_dir():
                files.append(f"[DIR]  {rel_path}/")
            else:
                size_kb = item.stat().st_size / 1024
                files.append(f"[FILE] {rel_path} ({size_kb:.1f}KB)")
        
        return "\n".join(files) if files else "Directory is empty"
    
    def _tool_search_files(self, pattern: str, directory: str = ".") -> str:
        """Search for files by pattern"""
        dir_path = self.workspace_dir / directory
        
        if not self._is_safe_path(str(dir_path)):
            raise PermissionError(f"Access denied to directory: {directory}")
        
        matches = []
        for path in dir_path.rglob(pattern):
            if self._is_safe_path(str(path)):
                rel_path = path.relative_to(self.workspace_dir)
                matches.append(str(rel_path))
        
        return "\n".join(matches) if matches else "No matches found"
    
    def _tool_execute_code(self, code: str) -> str:
        """Execute Python code"""
        if not self.config.enable_code_execution:
            raise PermissionError("Code execution is disabled")
        
        # Create a temporary file
        temp_file = self.workspace_dir / f"_temp_exec_{int(time.time())}.py"
        temp_file.write_text(code)
        
        try:
            result = subprocess.run(
                ["python3", str(temp_file)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.workspace_dir)
            )
            
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"
            
            return output or "Code executed successfully (no output)"
        
        finally:
            temp_file.unlink()
    
    def _tool_run_command(self, command: str) -> str:
        """Run a shell command"""
        if not self.config.enable_code_execution:
            raise PermissionError("Command execution is disabled")
        
        if self.config.safe_execution_only:
            # Whitelist of safe commands
            safe_commands = ["ls", "pwd", "echo", "cat", "grep", "find", "wc", "head", "tail"]
            cmd_name = command.split()[0]
            if cmd_name not in safe_commands:
                raise PermissionError(f"Command not in whitelist: {cmd_name}")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.workspace_dir)
            )
            
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"
            
            return output or "Command executed successfully (no output)"
        
        except subprocess.TimeoutExpired:
            return "Command timed out after 30 seconds"
    
    def _tool_read_system_prompt(self) -> str:
        """Read the current system prompt"""
        return self.system_prompt
    
    def _tool_edit_system_prompt(self, new_prompt: str) -> str:
        """Edit the system prompt"""
        self.system_prompt = new_prompt
        Path(self.config.system_prompt_file).write_text(new_prompt)
        return "System prompt updated successfully"
    
    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """Parse tool calls from AI response"""
        tool_calls = []
        
        # Find all tool call blocks
        pattern = r'<tool_call>\s*<tool_name>(.*?)</tool_name>\s*<parameters>(.*?)</parameters>\s*</tool_call>'
        matches = re.finditer(pattern, response, re.DOTALL)
        
        for match in matches:
            tool_name = match.group(1).strip()
            params_str = match.group(2).strip()
            
            try:
                parameters = json.loads(params_str)
                tool_calls.append({
                    "tool_name": tool_name,
                    "parameters": parameters
                })
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse parameters for tool {tool_name}")
        
        return tool_calls
    
    def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute parsed tool calls"""
        results = []
        
        for call in tool_calls:
            tool_name = call["tool_name"]
            parameters = call["parameters"]
            
            if tool_name not in self.tools:
                results.append({
                    "tool_name": tool_name,
                    "success": False,
                    "error": f"Unknown tool: {tool_name}"
                })
                continue
            
            print(f"  Executing tool: {tool_name}")
            result = self.tools[tool_name].execute(**parameters)
            result["tool_name"] = tool_name
            results.append(result)
        
        return results
    
    def _format_tool_results(self, results: List[Dict[str, Any]]) -> str:
        """Format tool results for the AI"""
        if not results:
            return ""
        
        formatted = "\n\nTool Results:\n"
        for result in results:
            formatted += f"\n[{result['tool_name']}]\n"
            if result["success"]:
                formatted += f"Success: {result['result']}\n"
            else:
                formatted += f"Error: {result['error']}\n"
        
        return formatted
    
    def run_iteration(self, user_message: Optional[str] = None) -> str:
        """Run a single iteration of the agent"""
        print(f"\n{'='*60}")
        print(f"Agent Iteration - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
        # Add user message if provided
        if user_message:
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })
            print(f"User: {user_message}\n")
        else:
            # Auto-continue from previous conversation
            self.conversation_history.append({
                "role": "user",
                "content": "Continue with your current task. What's the next step?"
            })
        
        # Generate AI response
        print("Generating AI response...")
        response = self.backend.generate_response(
            self.conversation_history,
            self.system_prompt
        )
        
        print(f"\nAgent: {response}\n")
        
        # Parse and execute tool calls
        tool_calls = self._parse_tool_calls(response)
        
        if tool_calls:
            print(f"Found {len(tool_calls)} tool call(s)")
            results = self._execute_tool_calls(tool_calls)
            tool_results_text = self._format_tool_results(results)
            response += tool_results_text
            print(tool_results_text)
        
        # Add response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        # Save conversation
        self._save_conversation_history()
        
        return response
    
    def run_loop(self):
        """Run the agent in a continuous loop"""
        print(f"\n{'#'*60}")
        print("Autonomous Agent Started")
        print(f"Backend: {self.config.ai_backend}")
        print(f"Model: {self.config.model}")
        print(f"Run Interval: {self.config.run_interval_seconds} seconds")
        print(f"Workspace: {self.workspace_dir}")
        print(f"{'#'*60}\n")
        
        try:
            while True:
                self.run_iteration()
                print(f"\nWaiting {self.config.run_interval_seconds} seconds until next iteration...")
                print("(Press Ctrl+C to stop)\n")
                time.sleep(self.config.run_interval_seconds)
        
        except KeyboardInterrupt:
            print("\n\nAgent stopped by user")
            self._save_conversation_history()
    
    def run_interactive(self):
        """Run the agent in interactive mode"""
        print(f"\n{'#'*60}")
        print("Autonomous Agent - Interactive Mode")
        print(f"Backend: {self.config.ai_backend}")
        print(f"Model: {self.config.model}")
        print(f"Workspace: {self.workspace_dir}")
        print(f"{'#'*60}\n")
        print("Type 'quit' to exit, 'auto' to switch to auto mode\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                
                if user_input.lower() == 'auto':
                    print("\nSwitching to autonomous mode...")
                    self.run_loop()
                    break
                
                if user_input:
                    self.run_iteration(user_input)
            
            except KeyboardInterrupt:
                print("\n\nAgent stopped by user")
                break
        
        self._save_conversation_history()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Autonomous Coding Agent")
    parser.add_argument("--backend", default="gemini", choices=["gemini", "anthropic", "openai"],
                       help="AI backend to use")
    parser.add_argument("--api-key", required=True, help="API key for the AI backend")
    parser.add_argument("--model", default="", help="Model name (uses default if not specified)")
    parser.add_argument("--interval", type=int, default=60, help="Run interval in seconds")
    parser.add_argument("--workspace", default="./workspace", help="Workspace directory")
    parser.add_argument("--mode", default="interactive", choices=["interactive", "auto"],
                       help="Run mode: interactive or autonomous")
    parser.add_argument("--disable-execution", action="store_true", help="Disable code execution")
    parser.add_argument("--unsafe-execution", action="store_true", help="Allow all commands (dangerous!)")
    
    args = parser.parse_args()
    
    # Set default models
    default_models = {
        "gemini": "gemini-1.5-flash",
        "anthropic": "claude-sonnet-4-5-20250929",
        "openai": "gpt-4"
    }
    model = args.model or default_models[args.backend]
    
    # Create config
    config = AgentConfig(
        ai_backend=args.backend,
        api_key=args.api_key,
        model=model,
        run_interval_seconds=args.interval,
        workspace_dir=args.workspace,
        enable_code_execution=not args.disable_execution,
        safe_execution_only=not args.unsafe_execution
    )
    
    # Create and run agent
    agent = AutonomousAgent(config)
    
    if args.mode == "interactive":
        agent.run_interactive()
    else:
        agent.run_loop()


if __name__ == "__main__":
    main()
