
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from aecp import AECP
from aecp.adapters import LocalModelAdapter
from sklearn.metrics.pairwise import cosine_similarity

def evaluate_aecp_fidelity():
    print("🚀 Initializing AECP Evaluation...")
    
    # 1. Load models
    print("Loading models (this might take a moment if not cached)...")
    model_a = SentenceTransformer('all-MiniLM-L6-v2') # 384 dim
    model_b = SentenceTransformer('all-mpnet-base-v2') # 768 dim
    
    # 2. Setup AECP Agents
    agent_a = AECP(LocalModelAdapter(model_a), agent_id="agent_minilm")
    agent_b = AECP(LocalModelAdapter(model_b), agent_id="agent_mpnet")
    
    # 3. Calibration
    print("\n🤝 Calibrating agents...")
    start_cal = time.time()
    calibration_result = agent_a.calibrate_with(agent_b)
    end_cal = time.time()
    
    print(f"✅ Calibration complete in {end_cal - start_cal:.2f}s")
    print(f"   Validation Similarity: {calibration_result.validation_similarity:.4f}")
    
    # 4. Evaluation
    test_phrases = [
        "The quick brown fox jumps over the lazy dog",
        "Artificial intelligence is transforming the world",
        "How do I cook a perfect steak?",
        "Quantum computing relies on superposition and entanglement",
        "The stock market is volatile today",
        "I love the smell of rain on hot pavement",
        "Developing software requires patience and discipline",
        "The capital of France is Paris",
        "AECP enables seamless embedding transfer between agents",
        "Vector databases are essential for RAG applications"
    ]
    
    similarities_aecp = []
    
    print("\n📊 Evaluating Semantic Fidelity...")
    print(f"{'Phrase':<50} | {'AECP Fidelity':<15}")
    print("-" * 70)
    
    for phrase in test_phrases:
        # Get ground truth embedding from Agent B
        truth_b = agent_b.embed(phrase)
        
        # Transfer from Agent A to Agent B space
        transfer_result = agent_a.transfer_to(agent_b.agent_id, phrase)
        transferred_b = transfer_result.embedding
        
        # Calculate cosine similarity
        sim = cosine_similarity(
            np.array(truth_b).reshape(1, -1),
            np.array(transferred_b).reshape(1, -1)
        )[0][0]
        
        similarities_aecp.append(sim)
        print(f"{phrase[:47]+'...':<50} | {sim:.4%}")
    
    avg_sim = np.mean(similarities_aecp)
    print("-" * 70)
    print(f"{'AVERAGE FIDELITY':<50} | {avg_sim:.4%}")
    
    # 5. Conclusion
    print("\n💡 Analysis:")
    if avg_sim > 0.95:
        print("Superior Performance: AECP preserved >95% of the semantic meaning across different model architectures.")
    else:
        print(f"Good Performance: AECP preserved {avg_sim:.1%} of semantic meaning.")
    
    print("\nComparison to Text-Only:")
    print("- Text-only: Requires target model to re-encode (Loss of context if source used proprietary features)")
    print("- AECP: Direct vector mapping, bypasses re-encoding if the source already had the vector.")

if __name__ == "__main__":
    evaluate_aecp_fidelity()
