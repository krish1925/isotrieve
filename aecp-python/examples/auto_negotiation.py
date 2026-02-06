"""
AECP Auto-Negotiation Example

This example demonstrates how AECP automatically negotiates between agents:
1. When both agents support AECP -> Uses AECP
2. When only one agent supports AECP -> Falls back to text
3. When neither agent supports AECP -> Uses text
"""

from aecp import AECP, AECPNegotiator
from aecp.adapters import MockAdapter


def main():
    print("="*70)
    print("AECP Auto-Negotiation Demo")
    print("="*70)
    
    # Scenario 1: Both agents support AECP
    print("\n\n📋 Scenario 1: Both agents support AECP")
    print("-" * 70)
    
    agent_a = AECP(MockAdapter(dimensions=384), agent_id="agent_a")
    agent_b = AECP(MockAdapter(dimensions=768), agent_id="agent_b")
    
    method = AECPNegotiator.negotiate(agent_a, agent_b, verbose=True)
    
    if method.uses_aecp:
        print(f"✓ Using AECP with {method.calibration_result.validation_similarity:.1%} fidelity")
        
        # Send a message using AECP
        message = "Hello, how are you?"
        result = AECPNegotiator.send_message(agent_a, agent_b, message, method=method)
        print(f"\n📤 Sent via AECP: '{message}'")
        print(f"   Transfer ID: {result['transfer_id']}")
        print(f"   Expected similarity: {result['expected_similarity']:.1%}")
    
    # Scenario 2: Only one agent supports AECP
    print("\n\n📋 Scenario 2: Only one agent supports AECP")
    print("-" * 70)
    
    agent_aecp = AECP(MockAdapter(dimensions=384), agent_id="agent_aecp")
    agent_plain = {"name": "PlainAgent", "type": "non-aecp"}  # Just a regular object
    
    method = AECPNegotiator.negotiate(agent_aecp, agent_plain, verbose=True)
    
    if not method.uses_aecp:
        print(f"✓ Using text fallback")
        print(f"   Reason: {method.fallback_reason}")
        
        # Send a message using text
        message = "Hello, I don't support AECP"
        result = AECPNegotiator.send_message(agent_aecp, agent_plain, message, method=method)
        print(f"\n📤 Sent via text: '{result['message']}'")
    
    # Scenario 3: Neither agent supports AECP
    print("\n\n📋 Scenario 3: Neither agent supports AECP")
    print("-" * 70)
    
    agent1_plain = {"name": "Agent1", "type": "non-aecp"}
    agent2_plain = {"name": "Agent2", "type": "non-aecp"}
    
    method = AECPNegotiator.negotiate(agent1_plain, agent2_plain, verbose=True)
    
    if not method.uses_aecp:
        print(f"✓ Using text fallback")
        print(f"   Reason: {method.fallback_reason}")
    
    # Scenario 4: Demonstrate automatic re-negotiation
    print("\n\n📋 Scenario 4: Automatic re-negotiation on each message")
    print("-" * 70)
    
    agent_x = AECP(MockAdapter(dimensions=512), agent_id="agent_x")
    agent_y = AECP(MockAdapter(dimensions=256), agent_id="agent_y")
    
    print("\nSending message without pre-negotiation (will auto-negotiate)...")
    result = AECPNegotiator.send_message(
        agent_x, 
        agent_y, 
        "This will trigger automatic negotiation",
        verbose=True
    )
    
    if result['method'] == 'aecp':
        print(f"✓ Auto-negotiated AECP successfully")
    
    # Summary
    print("\n\n" + "="*70)
    print("Summary")
    print("="*70)
    print("""
Key Features of Auto-Negotiation:

1. ✓ Automatic Detection
   - Detects if both agents support AECP
   - No manual configuration needed

2. ✓ Seamless Fallback
   - Falls back to text if one agent doesn't support AECP
   - Provides clear warning messages

3. ✓ Calibration on Demand
   - Automatically calibrates when both agents support AECP
   - Caches matrices for future use

4. ✓ Transparent Communication
   - Returns clear status about which method is being used
   - Provides fallback reasons when AECP is not available

Usage in Your Code:
------------------
from aecp import AECP, AECPNegotiator
from aecp.adapters import OpenAIAdapter

# Just create your agents normally
agent1 = AECP(OpenAIAdapter(api_key="..."))
agent2 = some_other_agent  # Could be AECP or not

# Auto-negotiate and send
result = AECPNegotiator.send_message(agent1, agent2, "Hello!")

# The library handles everything:
# - Checks if both support AECP
# - Calibrates if needed
# - Uses AECP or falls back to text
# - Returns result with method info
""")


if __name__ == "__main__":
    main()
