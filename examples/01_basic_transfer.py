
"""
01_basic_transfer.py
--------------------
Demonstrates the core Isotrieve workflow:
1. Setup two agents with different models.
2. Calibrate them (learning the transfer matrix).
3. Transfer a vector from A to B.
"""

import sys
import os
# Add parent directory to path to import isotrieve if installed locally
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "isotrieve-python"))

from isotrieve import Isotrieve
from isotrieve.adapters import LocalModelAdapter
from sentence_transformers import SentenceTransformer
import numpy as np

def main():
    # 1. Initialize Agents
    # We use small models for the example to run fast.
    print("Initialize Agents...")
    agent_a = Isotrieve(LocalModelAdapter(SentenceTransformer('all-MiniLM-L6-v2')))
    agent_b = Isotrieve(LocalModelAdapter(SentenceTransformer('all-mpnet-base-v2')))

    # 2. Calibrate
    # In production, you would save this matrix and load it later.
    print("Calibrating (Learning Transfer Matrix)...")
    agent_a.calibrate_with(agent_b)

    # 3. The Transfer
    print("\n--- Transfer Scenario ---")
    query = "The server is returning 500 errors on the login endpoint."
    
    # A's internal representation
    vector_a = agent_a.embed(query)
    
    # Transform to B's space
    vector_b_transferred = agent_a.transfer_to(agent_b, vector_a)
    
    # 4. Verify
    # Let's see if B understands it.
    vector_b_native = agent_b.embed(query)
    
    # Compare
    from sklearn.metrics.pairwise import cosine_similarity
    
    similarity = cosine_similarity(
        [vector_b_transferred], 
        [vector_b_native]
    )[0][0]
    
    print(f"Original Text: '{query}'")
    print(f"Fidelity Score: {similarity:.4f}")
    
    if similarity > 0.95:
        print("Success! The meaning was preserved perfectly.")
    else:
        print("Transfer complete (with some signal loss).")

if __name__ == "__main__":
    main()
