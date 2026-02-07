
import sys
import os
import time

# Add parent directory to path to import aecp if installed locally
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "aecp-python"))

try:
    from aecp import AECP
    from aecp.adapters import LocalModelAdapter
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Please install requirements: pip install sentence-transformers aecp")
    sys.exit(1)

def type_effect(text, delay=0.03):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def main():
    print("\n AECP: Agent Embedding Communication Protocol")
    print("---------------------------------------------")
    type_effect("Initializing Agent A (Source)...")
    model_a_name = 'all-MiniLM-L6-v2'
    agent_a = AECP(LocalModelAdapter(SentenceTransformer(model_a_name)))
    print(f" Agent A ready ({model_a_name})")

    type_effect("Initializing Agent B (Target)...")
    model_b_name = 'all-mpnet-base-v2'
    agent_b = AECP(LocalModelAdapter(SentenceTransformer(model_b_name)))
    print(f" Agent B ready ({model_b_name})")

    print("\n Requesting Calibration...")
    time.sleep(0.5)
    agent_a.calibrate_with(agent_b)
    print(" Transfer Matrix Established.")

    while True:
        print("\n" + "="*50)
        query = input("ENTER A PHRASE for Agent A (or 'exit'): ")
        if query.lower() in ('exit', 'quit'):
            break

        # 1. Agent A encodes
        vec_a = agent_a.embed(query)
        print(f"\n[Agent A] Encoded into {len(vec_a)} dims.")

        # 2. Transfer
        start_t = time.time()
        vec_transferred = agent_a.transfer_to(agent_b, vec_a)
        duration = (time.time() - start_t) * 1000
        print(f"[AECP] Transferred to Agent B space in {duration:.2f}ms ")

        # 3. Validation (Agent B checks against its own truth)
        vec_truth = agent_b.embed(query)
        
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        sim = cosine_similarity([vec_transferred], [vec_truth])[0][0]
        
        print(f"[Agent B] Received vector. Compared to my own understanding:")
        print(f"          Semantic Fidelity: {sim:.2%}")
        
        if sim > 0.9:
             print("          Result:  PERFECT MATCH")
        else:
             print("          Result: ⚠️  LOSS DETECTED")

if __name__ == "__main__":
    main()
