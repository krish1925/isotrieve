"""
Isotrieve Auto-Negotiation Example

This example demonstrates how Isotrieve automatically negotiates between agents:
1. When both agents support Isotrieve -> Uses Isotrieve
2. When only one agent supports Isotrieve -> Falls back to text
3. When neither agent supports Isotrieve -> Uses text
"""

from isotrieve import Isotrieve, IsotrieveNegotiator
from isotrieve.adapters import MockAdapter


def main():
    print("="*70)
    print("Isotrieve Auto-Negotiation Demo")
    print("="*70)
    
    # Scenario 1: Both agents support Isotrieve
    print("\n\n Scenario 1: Both agents support Isotrieve")
    print("-" * 70)
    
    agent_a = Isotrieve(MockAdapter(dimensions=384), agent_id="agent_a")
    agent_b = Isotrieve(MockAdapter(dimensions=768), agent_id="agent_b")
    
    method = IsotrieveNegotiator.negotiate(agent_a, agent_b, verbose=True)
    
    if method.uses_isotrieve:
        print(f"✓ Using Isotrieve with {method.calibration_result.validation_similarity:.1%} fidelity")
        
        # Send a message using Isotrieve
        message = "Hello, how are you?"
        result = IsotrieveNegotiator.send_message(agent_a, agent_b, message, method=method)
        print(f"\n Sent via Isotrieve: '{message}'")
        print(f"   Transfer ID: {result['transfer_id']}")
        print(f"   Expected similarity: {result['expected_similarity']:.1%}")
    
    # Scenario 2: Only one agent supports Isotrieve
    print("\n\n Scenario 2: Only one agent supports Isotrieve")
    print("-" * 70)
    
    agent_isotrieve = Isotrieve(MockAdapter(dimensions=384), agent_id="agent_isotrieve")
    agent_plain = {"name": "PlainAgent", "type": "non-isotrieve"}  # Just a regular object
    
    method = IsotrieveNegotiator.negotiate(agent_isotrieve, agent_plain, verbose=True)
    
    if not method.uses_isotrieve:
        print(f"✓ Using text fallback")
        print(f"   Reason: {method.fallback_reason}")
        
        # Send a message using text
        message = "Hello, I don't support Isotrieve"
        result = IsotrieveNegotiator.send_message(agent_isotrieve, agent_plain, message, method=method)
        print(f"\n Sent via text: '{result['message']}'")
    
    # Scenario 3: Neither agent supports Isotrieve
    print("\n\n Scenario 3: Neither agent supports Isotrieve")
    print("-" * 70)
    
    agent1_plain = {"name": "Agent1", "type": "non-isotrieve"}
    agent2_plain = {"name": "Agent2", "type": "non-isotrieve"}
    
    method = IsotrieveNegotiator.negotiate(agent1_plain, agent2_plain, verbose=True)
    
    if not method.uses_isotrieve:
        print(f"✓ Using text fallback")
        print(f"   Reason: {method.fallback_reason}")
    
    # Scenario 4: Demonstrate automatic re-negotiation
    print("\n\n Scenario 4: Automatic re-negotiation on each message")
    print("-" * 70)
    
    agent_x = Isotrieve(MockAdapter(dimensions=512), agent_id="agent_x")
    agent_y = Isotrieve(MockAdapter(dimensions=256), agent_id="agent_y")
    
    print("\nSending message without pre-negotiation (will auto-negotiate)...")
    result = IsotrieveNegotiator.send_message(
        agent_x, 
        agent_y, 
        "This will trigger automatic negotiation",
        verbose=True
    )
    
    if result['method'] == 'isotrieve':
        print(f"✓ Auto-negotiated Isotrieve successfully")
    
    # Summary
    print("\n\n" + "="*70)
    print("Summary")
    print("="*70)
    print("""
Key Features of Auto-Negotiation:

1. ✓ Automatic Detection
   - Detects if both agents support Isotrieve
   - No manual configuration needed

2. ✓ Seamless Fallback
   - Falls back to text if one agent doesn't support Isotrieve
   - Provides clear warning messages

3. ✓ Calibration on Demand
   - Automatically calibrates when both agents support Isotrieve
   - Caches matrices for future use

4. ✓ Transparent Communication
   - Returns clear status about which method is being used
   - Provides fallback reasons when Isotrieve is not available

Usage in Your Code:
------------------
from isotrieve import Isotrieve, IsotrieveNegotiator
from isotrieve.adapters import OpenAIAdapter

# Just create your agents normally
agent1 = Isotrieve(OpenAIAdapter(api_key="..."))
agent2 = some_other_agent  # Could be Isotrieve or not

# Auto-negotiate and send
result = IsotrieveNegotiator.send_message(agent1, agent2, "Hello!")

# The library handles everything:
# - Checks if both support Isotrieve
# - Calibrates if needed
# - Uses Isotrieve or falls back to text
# - Returns result with method info
""")


if __name__ == "__main__":
    main()
