# Autonomous Coding Agent - Project Summary

## 🎯 What You've Got

A complete, production-ready autonomous coding agent framework that rivals Cursor but with more flexibility and control. Built from scratch with your requirements in mind.

## 📦 Project Files

### Core Components
1. **autonomous_agent.py** - Main agent with all core functionality
   - Multiple AI backends (Gemini, Claude, GPT)
   - File operations (read, write, edit)
   - Code execution
   - System prompt editing
   - Configurable run frequency
   - Safety controls

2. **enhanced_agent.py** - Enhanced version with monitoring
   - Everything from basic agent
   - Logging to files
   - Cost tracking
   - Statistics monitoring
   - Better error handling

3. **quick_start.py** - Interactive setup wizard
   - Guides you through configuration
   - No command line needed
   - Perfect for first-time users

4. **run_agent.sh** - Shell launcher script
   - Easy one-command startup
   - Multiple launch modes
   - Automatic dependency checking

5. **examples.py** - 10+ usage examples
   - Different use cases
   - Programmatic control examples
   - Best practices

### Configuration
6. **requirements.txt** - Python dependencies
7. **config.example.json** - Example configuration
8. **README.md** - Complete documentation

## 🚀 Quick Start (3 Ways)

### Option 1: Easiest (Shell Launcher)
```bash
./run_agent.sh
```
Follow the prompts!

### Option 2: Quick Start Script
```bash
python3 quick_start.py
```
Guided setup with no command line arguments.

### Option 3: Direct Command
```bash
# Interactive mode
python3 autonomous_agent.py --backend gemini --api-key YOUR_KEY

# Autonomous mode (runs every 60 seconds)
python3 autonomous_agent.py --backend gemini --api-key YOUR_KEY --mode auto --interval 60
```

## 🔧 Key Features Implemented

### ✅ Multiple AI Backends
- Google Gemini (cheapest/free)
- Anthropic Claude (most capable)
- OpenAI GPT (balanced)

### ✅ File Operations
- Read files
- Write new files
- Edit specific lines
- List directories
- Search for files
- All with safety restrictions

### ✅ Code Execution
- Run Python code
- Execute shell commands
- Whitelisted safe commands by default
- Can enable unrestricted (for advanced use)

### ✅ Tunable Frequency
- Run every 30 seconds for fast iteration
- Run every 1-5 minutes for deliberate work
- Custom intervals (any number of seconds)
- Can pause/resume in interactive mode

### ✅ System Prompt Control
- Agent can read its own prompt
- Agent can edit its own prompt
- You can manually edit before starting
- Enables self-improvement

### ✅ Safety Features
- Workspace restrictions (can't access system files)
- File extension whitelist
- File size limits
- Blocked path protections
- Command whitelisting
- Can disable code execution entirely

### ✅ Two Modes
- **Interactive**: Chat with the agent, guide it
- **Autonomous**: Agent runs on its own at set intervals

## 💡 Usage Examples

### Example 1: Build a Web API
```bash
python3 autonomous_agent.py --backend gemini --api-key YOUR_KEY
```
Then say:
```
Create a Flask REST API with user management endpoints
```

### Example 2: Continuous Refactoring
```bash
python3 autonomous_agent.py --backend gemini --api-key YOUR_KEY --mode auto --interval 120
```
Tell it:
```
Refactor all Python files in the workspace to follow best practices
```
It will work continuously, checking progress every 2 minutes.

### Example 3: Fast Prototyping
```bash
python3 autonomous_agent.py --backend gemini --api-key YOUR_KEY --mode auto --interval 30
```
Tell it:
```
Create 10 different data structure implementations with tests
```
Runs every 30 seconds for rapid iteration.

## 🎛️ Configuration Options

### Run Frequency
- `--interval 30` - Every 30 seconds
- `--interval 60` - Every minute (default)
- `--interval 300` - Every 5 minutes
- `--interval 900` - Every 15 minutes

### AI Backend
- `--backend gemini` - Google Gemini (free!)
- `--backend anthropic` - Claude (best quality)
- `--backend openai` - GPT (balanced)

### Safety
- `--disable-execution` - No code execution
- `--unsafe-execution` - Allow all commands (⚠️ dangerous)
- Default is safe whitelisted commands only

### Workspace
- `--workspace ./my_project` - Custom directory
- Default is `./workspace`

## 📊 Enhanced Features (enhanced_agent.py)

The enhanced version adds:
- **Logging**: All actions logged to `./logs/`
- **Cost Tracking**: Approximate API costs
- **Statistics**: Track files created, edits, executions
- **Periodic Reports**: Stats every 5 iterations
- **Better Error Handling**: Graceful degradation

Use it the same way:
```bash
python3 enhanced_agent.py --backend gemini --api-key YOUR_KEY
```

## 🔐 API Keys

### Get Your Keys
- **Gemini**: https://makersuite.google.com/app/apikey (FREE!)
- **Claude**: https://console.anthropic.com/ (pay as you go)
- **OpenAI**: https://platform.openai.com/api-keys (pay as you go)

### Set as Environment Variables (Recommended)
```bash
export GEMINI_API_KEY="your_key_here"
export ANTHROPIC_API_KEY="your_key_here"
export OPENAI_API_KEY="your_key_here"
```

Then you don't need `--api-key` flag.

## 🏗️ How It Works

### Architecture
```
User Input
    ↓
AI Model (Gemini/Claude/GPT)
    ↓
Response with <tool_call> blocks
    ↓
Tool Execution (read_file, write_file, etc.)
    ↓
Results fed back to AI
    ↓
Continue or wait for next interval
```

### Tool Format
The AI uses XML tags to call tools:
```xml
<tool_call>
<tool_name>write_file</tool_name>
<parameters>
{
  "filepath": "app.py",
  "content": "print('Hello World')"
}
</parameters>
</tool_call>
```

### Available Tools
1. `read_file` - Read a file
2. `write_file` - Create/overwrite a file
3. `edit_file` - Edit specific lines
4. `list_files` - List directory contents
5. `search_files` - Find files by pattern
6. `execute_code` - Run Python code
7. `run_command` - Run shell commands
8. `read_system_prompt` - View current prompt
9. `edit_system_prompt` - Modify the prompt

## 🆚 Comparison to Cursor

| Feature | This Agent | Cursor |
|---------|-----------|--------|
| Multiple AI backends | ✅ | ❌ |
| Autonomous mode | ✅ | ❌ |
| Tunable frequency | ✅ | N/A |
| System prompt access | ✅ | ❌ |
| Open source / customizable | ✅ | ❌ |
| Cost tracking | ✅ | ❌ |
| IDE integration | ❌ | ✅ |
| GUI | ❌ | ✅ |
| Free tier | ✅ (Gemini) | Limited |

## 🎨 Customization

### 1. Add New Tools
Edit `autonomous_agent.py`, add to `_register_tools()`:
```python
tools["my_tool"] = Tool(
    "my_tool",
    "Description",
    self._tool_my_implementation
)
```

### 2. Custom System Prompt
Edit `system_prompt.txt` before starting, or:
```
Edit your system prompt to focus on testing and documentation
```

### 3. Change Safety Rules
In the code, modify:
- `allowed_file_extensions`
- `blocked_paths`
- `max_file_size_kb`
- Safe command whitelist

## 📈 Best Practices

1. **Start Interactive**: Always begin in interactive mode to guide the agent
2. **Be Specific**: Give clear, detailed initial instructions
3. **Monitor Progress**: Check the workspace frequently
4. **Adjust Interval**: Start slow (2-5 minutes), speed up as needed
5. **Use Enhanced Version**: For production work, use enhanced_agent.py
6. **Keep Safe Mode**: Don't disable execution safety unless necessary
7. **Set Budgets**: Watch API costs, especially with Claude/GPT

## 🔒 Security Notes

⚠️ **Important**:
1. Never use `--unsafe-execution` with untrusted code
2. Keep workspace separate from important files
3. Don't expose API keys
4. Review changes before deploying
5. Monitor the logs directory
6. Use safe mode for public/shared environments

## 🐛 Troubleshooting

**Agent stops responding:**
- Check API key validity
- Verify API rate limits
- Check conversation_history.json

**Permission errors:**
- Verify workspace exists
- Check file extensions allowed
- Ensure paths not in blocked list

**Code won't execute:**
- Check execution not disabled
- Verify command in whitelist
- Try unsafe mode carefully

## 💰 Cost Estimates

Using Gemini: **FREE** (generous free tier)
Using Claude: ~$0.01-0.05 per iteration
Using GPT-4: ~$0.02-0.10 per iteration

The enhanced_agent.py tracks costs automatically.

## 🚦 Next Steps

1. Install dependencies: `pip install -r requirements.txt`
2. Get an API key (Gemini is free!)
3. Run: `./run_agent.sh` or `python3 quick_start.py`
4. Give it a task!
5. Watch it work autonomously

## 📝 Example Session

```bash
$ python3 autonomous_agent.py --backend gemini --api-key YOUR_KEY

You: Create a simple calculator module with tests

Agent: I'll create a calculator module with comprehensive tests.

[Executing tool: write_file]
Created: calculator.py

[Executing tool: write_file]
Created: test_calculator.py

[Executing tool: execute_code]
Running tests... All tests passed!

You: auto

Switching to autonomous mode...
Running every 60 seconds...
```

## 🎓 Learning Resources

Check out:
- `examples.py` - 10 different usage patterns
- `README.md` - Full documentation
- Tool implementations in `autonomous_agent.py`
- Enhanced features in `enhanced_agent.py`

## ✨ What Makes This Special

Unlike Cursor, this agent:
- Runs **autonomously** at configurable intervals
- Can use **any AI backend** you want
- Can **edit its own system prompt**
- Tracks **costs and statistics**
- Is **completely open source**
- Has **no subscription required**
- Works **entirely from terminal**

Perfect for:
- Automated refactoring sessions
- Continuous code improvement
- Background task automation
- Learning AI agent development
- Running on servers/cloud
- Custom workflows

## 🎉 You're Ready!

You have a complete, production-ready autonomous coding agent. It's like Cursor, but with more control, more flexibility, and the ability to run truly autonomously.

Start with interactive mode, give it a task, and watch it work!

```bash
./run_agent.sh
```

Good luck! 🚀
