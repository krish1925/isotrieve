#!/usr/bin/env python3
"""
Enhanced Autonomous Agent with logging, monitoring, and cost tracking
"""

import logging
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, Any
from autonomous_agent import AutonomousAgent, AgentConfig


class EnhancedAgent(AutonomousAgent):
    """Enhanced agent with additional monitoring and logging features"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        
        # Set up logging
        self.log_dir = Path("./logs")
        self.log_dir.mkdir(exist_ok=True)
        
        log_file = self.log_dir / f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Stats tracking
        self.stats = {
            "iterations": 0,
            "tool_calls": 0,
            "files_created": 0,
            "files_edited": 0,
            "code_executions": 0,
            "errors": 0,
            "start_time": datetime.now().isoformat()
        }
        
        # Cost tracking (approximate)
        self.cost_per_1k_tokens = {
            "gemini": {"input": 0.00, "output": 0.00},  # Free tier
            "anthropic": {"input": 0.003, "output": 0.015},  # Claude Sonnet
            "openai": {"input": 0.01, "output": 0.03}  # GPT-4
        }
        
        self.estimated_cost = 0.0
        
        self.logger.info(f"Enhanced Agent initialized with {config.ai_backend} backend")
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token)"""
        return len(text) // 4
    
    def _track_cost(self, input_text: str, output_text: str):
        """Track approximate API costs"""
        input_tokens = self._estimate_tokens(input_text)
        output_tokens = self._estimate_tokens(output_text)
        
        costs = self.cost_per_1k_tokens.get(self.config.ai_backend, {"input": 0, "output": 0})
        
        cost = (input_tokens / 1000 * costs["input"] + 
                output_tokens / 1000 * costs["output"])
        
        self.estimated_cost += cost
        
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost
        }
    
    def run_iteration(self, user_message: str = None) -> str:
        """Enhanced iteration with logging and tracking"""
        self.stats["iterations"] += 1
        self.logger.info(f"Starting iteration #{self.stats['iterations']}")
        
        try:
            # Build input for cost tracking
            input_text = self.system_prompt + "\n"
            for msg in self.conversation_history:
                input_text += msg["content"] + "\n"
            
            # Run the actual iteration
            response = super().run_iteration(user_message)
            
            # Track cost
            cost_info = self._track_cost(input_text, response)
            self.logger.info(f"Iteration cost: ${cost_info['cost']:.4f} "
                           f"(~{cost_info['input_tokens']} in, ~{cost_info['output_tokens']} out)")
            
            # Track tool calls
            tool_calls = self._parse_tool_calls(response)
            self.stats["tool_calls"] += len(tool_calls)
            
            for call in tool_calls:
                tool_name = call["tool_name"]
                self.logger.info(f"Tool called: {tool_name}")
                
                # Update specific stats
                if tool_name == "write_file":
                    self.stats["files_created"] += 1
                elif tool_name == "edit_file":
                    self.stats["files_edited"] += 1
                elif tool_name == "execute_code":
                    self.stats["code_executions"] += 1
            
            self.logger.info("Iteration completed successfully")
            return response
            
        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error(f"Iteration failed: {str(e)}", exc_info=True)
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        stats = self.stats.copy()
        stats["estimated_total_cost"] = self.estimated_cost
        stats["current_time"] = datetime.now().isoformat()
        return stats
    
    def save_stats(self):
        """Save statistics to file"""
        stats_file = self.log_dir / "stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.get_stats(), f, indent=2)
        self.logger.info(f"Stats saved to {stats_file}")
    
    def print_stats(self):
        """Print current statistics"""
        stats = self.get_stats()
        
        print("\n" + "="*60)
        print("Agent Statistics")
        print("="*60)
        print(f"Iterations: {stats['iterations']}")
        print(f"Tool Calls: {stats['tool_calls']}")
        print(f"Files Created: {stats['files_created']}")
        print(f"Files Edited: {stats['files_edited']}")
        print(f"Code Executions: {stats['code_executions']}")
        print(f"Errors: {stats['errors']}")
        print(f"Estimated Cost: ${stats['estimated_total_cost']:.4f}")
        print("="*60 + "\n")
    
    def run_loop(self):
        """Enhanced loop with periodic stats"""
        self.logger.info("Starting autonomous loop")
        
        try:
            iteration_count = 0
            while True:
                self.run_iteration()
                iteration_count += 1
                
                # Print stats every 5 iterations
                if iteration_count % 5 == 0:
                    self.print_stats()
                    self.save_stats()
                
                print(f"\nWaiting {self.config.run_interval_seconds} seconds...")
                print("(Press Ctrl+C to stop)\n")
                time.sleep(self.config.run_interval_seconds)
        
        except KeyboardInterrupt:
            self.logger.info("Agent stopped by user")
            self.print_stats()
            self.save_stats()
            print("\n\nAgent stopped")
    
    def run_interactive(self):
        """Enhanced interactive mode"""
        self.logger.info("Starting interactive mode")
        
        print(f"\n{'#'*60}")
        print("Enhanced Autonomous Agent - Interactive Mode")
        print(f"{'#'*60}\n")
        print("Commands:")
        print("  quit - Exit the agent")
        print("  auto - Switch to autonomous mode")
        print("  stats - Show current statistics")
        print("  help - Show this help message")
        print()
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                
                if user_input.lower() == 'auto':
                    print("\nSwitching to autonomous mode...")
                    self.run_loop()
                    break
                
                if user_input.lower() == 'stats':
                    self.print_stats()
                    continue
                
                if user_input.lower() == 'help':
                    print("\nCommands:")
                    print("  quit - Exit")
                    print("  auto - Switch to auto mode")
                    print("  stats - Show statistics")
                    print("  help - This message\n")
                    continue
                
                if user_input:
                    self.run_iteration(user_input)
            
            except KeyboardInterrupt:
                self.logger.info("Agent stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Error: {str(e)}")
                print(f"\n❌ Error: {str(e)}\n")
        
        self.print_stats()
        self.save_stats()


def main():
    """Main entry point for enhanced agent"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Autonomous Coding Agent")
    parser.add_argument("--backend", default="gemini", choices=["gemini", "anthropic", "openai"])
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--workspace", default="./workspace")
    parser.add_argument("--mode", default="interactive", choices=["interactive", "auto"])
    parser.add_argument("--disable-execution", action="store_true")
    parser.add_argument("--unsafe-execution", action="store_true")
    
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
    
    # Create and run enhanced agent
    agent = EnhancedAgent(config)
    
    if args.mode == "interactive":
        agent.run_interactive()
    else:
        agent.run_loop()


if __name__ == "__main__":
    import time  # Import needed for sleep
    main()
