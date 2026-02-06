# Autonomous Coding Agent

A lightweight, customizable autonomous coding agent that can edit files, execute code, and work continuously on programming tasks. Supports multiple AI backends (Gemini, Claude, GPT).

## Features

- **Multiple AI Backends**: Use Google Gemini, Anthropic Claude, or OpenAI GPT
- **File Operations**: Read, write, and edit files with safety restrictions
- **Code Execution**: Run Python code and shell commands locally
- **Configurable Run Frequency**: Set how often the agent runs (e.g., every 1 minute)
- **System Prompt Editing**: Agent can modify its own system prompt
- **Conversation Memory**: Maintains conversation history across sessions
- **Safety Controls**: Configurable file access restrictions and command whitelisting
- **Two Modes**: Interactive mode or fully autonomous mode

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your API key for your chosen backend:
   - **Gemini**: Get from https://makersuite.google.com/app/apikey
   - **Anthropic**: Get from https://console.anthropic.com/
   - **OpenAI**: Get from https://platform.openai.com/api-keys

## Quick Start

### Interactive Mode (Recommended for first use)

```bash
# Using Gemini
python autonomous_agent.py --backend gemini --api-key YOUR_API_KEY

# Using Claude
python autonomous_agent.py --backend anthropic --api-key YOUR_API_KEY

# Using GPT
python autonomous_agent.py --backend openai --api-key YOUR_API_KEY
```

### Autonomous Mode

Run the agent continuously with automatic iterations:

```bash
# Run every 60 seconds (default)
python autonomous_agent.py --backend gemini --api-key YOUR_API_KEY --mode auto

# Run every 5 minutes
python autonomous_agent.py --backend gemini --api-key YOUR_API_KEY --mode auto --interval 300

# Run every 30 seconds
python autonomous_agent.py --backend gemini --api-key YOUR_API_KEY --mode auto --interval 30
```

## Usage Examples

### Example 1: Let the agent create a simple web server

```bash
python autonomous_agent.py --backend gemini --api-key YOUR_KEY
```

Then type:
```
Create a simple Flask web server with a homepage and an API endpoint that returns JSON data
```

The agent will:
1. Create the necessary Python files
2. Write the Flask code
3. Optionally run it to test

### Example 2: Autonomous bug fixing

Start the agent in auto mode and give it an initial task:

```bash
python autonomous_agent.py --backend gemini --api-key YOUR_KEY --mode auto --interval 120
```

The agent will work continuously, checking its progress every 2 minutes.

### Example 3: Code refactoring

```
I have a messy Python script in my workspace. Please refactor it to follow PEP 8 standards, add docstrings, and improve the structure.
```

## Available Tools

The agent has access to these tools:

### File Operations
- `read_file`: Read file contents
- `write_file`: Write content to a file
- `edit_file`: Edit specific lines in a file
- `list_files`: List files in a directory
- `search_files`: Search for files by pattern

### Code Execution
- `execute_code`: Run Python code
- `run_command`: Run shell commands (whitelisted by default)

### Configuration
- `read_system_prompt`: View current system prompt
- `edit_system_prompt`: Modify the system prompt

## Configuration Options

### Command Line Arguments

```bash
--backend          AI backend: gemini, anthropic, or openai
--api-key          Your API key
--model            Specific model name (optional, uses defaults)
--interval         Seconds between auto runs (default: 60)
--workspace        Workspace directory (default: ./workspace)
--mode             interactive or auto (default: interactive)
--disable-execution Disable code execution
--unsafe-execution  Allow all commands (not recommended!)
```

### Safety Features

**File Access Restrictions:**
- Only allowed file extensions can be written (configured in code)
- Blocked paths prevent access to system directories
- File size limits prevent memory issues
- All paths are restricted to the workspace directory

**Code Execution Controls:**
- Can be completely disabled with `--disable-execution`
- Safe mode (default) only allows whitelisted commands:
  - `ls`, `pwd`, `echo`, `cat`, `grep`, `find`, `wc`, `head`, `tail`
- Unsafe mode (`--unsafe-execution`) allows all commands (use with caution!)
- 30-second timeout on all executions

## System Prompt Customization

The agent's behavior is controlled by its system prompt. You can:

1. Edit `system_prompt.txt` directly before starting
2. Ask the agent to modify its own prompt:
   ```
   Please update your system prompt to be more focused on testing and documentation
   ```

3. Use the `edit_system_prompt` tool programmatically

## Conversation History

The agent maintains conversation history in `conversation_history.json`. This allows:
- Continuity across sessions
- Context awareness over time
- Ability to resume previous tasks

The history is automatically trimmed to the most recent messages (configurable).

## Run Frequency Tuning

The `--interval` parameter controls how often the agent runs in autonomous mode:

```bash
# Every 30 seconds (fast iteration)
--interval 30

# Every 1 minute (default)
--interval 60

# Every 5 minutes (slower, more deliberate)
--interval 300

# Every 15 minutes
--interval 900
```

**Considerations:**
- Faster intervals use more API credits
- Slower intervals give the agent more time to think between actions
- For complex tasks, longer intervals may be better
- For simple repetitive tasks, shorter intervals work well

## Advanced Usage

### Using Specific Models

```bash
# Use Gemini Pro
python autonomous_agent.py --backend gemini --api-key YOUR_KEY --model gemini-1.5-pro

# Use Claude Opus
python autonomous_agent.py --backend anthropic --api-key YOUR_KEY --model claude-opus-4-5-20251101

# Use GPT-4 Turbo
python autonomous_agent.py --backend openai --api-key YOUR_KEY --model gpt-4-turbo
```

### Switching Modes

In interactive mode, you can switch to autonomous mode:
```
You: Let's automate this. Type 'auto' when ready.
auto
```

### Environment Variables

You can also set your API key as an environment variable:

```bash
export GEMINI_API_KEY="your_key_here"
export ANTHROPIC_API_KEY="your_key_here"
export OPENAI_API_KEY="your_key_here"
```

Then modify the script to read from these variables.

## Tips for Best Results

1. **Start Interactive**: Begin in interactive mode to guide the agent
2. **Be Specific**: Give clear, specific initial instructions
3. **Monitor Progress**: Check the workspace directory regularly
4. **Adjust Interval**: Start with longer intervals (2-5 minutes) for complex tasks
5. **Review System Prompt**: Customize it for your specific use case
6. **Safety First**: Keep `safe_execution_only` enabled unless you need specific commands

## Comparison to Cursor

| Feature | This Agent | Cursor |
|---------|-----------|--------|
| File Editing | ✅ | ✅ |
| Code Execution | ✅ | ✅ |
| Autonomous Mode | ✅ | ❌ |
| Multiple AI Backends | ✅ | ❌ |
| Configurable Run Frequency | ✅ | N/A |
| System Prompt Access | ✅ | ❌ |
| Workspace Restrictions | ✅ | ✅ |
| IDE Integration | ❌ | ✅ |
| GUI | ❌ | ✅ |

## Troubleshooting

**Agent stops responding:**
- Check your API key is valid
- Verify you haven't hit rate limits
- Check conversation_history.json for errors

**Permission errors:**
- Verify the workspace directory exists and is writable
- Check that files are in allowed_file_extensions
- Ensure paths aren't in blocked_paths

**Code execution fails:**
- Make sure execution isn't disabled
- Check if the command is in the whitelist (safe mode)
- Try with `--unsafe-execution` if needed (carefully!)

## Security Notes

⚠️ **Important Security Considerations:**

1. **Never run with `--unsafe-execution` on untrusted code**
2. **Review the workspace regularly** - the agent has write access
3. **Keep blocked_paths configured** to protect system directories
4. **Monitor API usage** - autonomous mode uses credits continuously
5. **Don't expose API keys** in code or config files
6. **Use a separate workspace** - don't point to important directories

## License

MIT License - feel free to modify and use as needed!

## Contributing

This is a minimal framework designed to be customized. Feel free to:
- Add new tools
- Improve safety features
- Add more AI backends
- Enhance the system prompt
- Add logging and monitoring

## Future Enhancements

Possible additions:
- Web interface for monitoring
- More sophisticated tool calling (function calling APIs)
- Git integration
- Project templates
- Multi-agent collaboration
- Cost tracking
- Task scheduling
- Rollback capabilities
