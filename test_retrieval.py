
import numpy as np
from sentence_transformers import SentenceTransformer
from aecp import AECP
from aecp.adapters import LocalModelAdapter
from aecp.matrix import cosine_similarity as aecp_sim

def test_retrieval_preservation():
    print("🚀 Loading models...")
    model_a = SentenceTransformer('all-MiniLM-L6-v2')
    model_b = SentenceTransformer('all-mpnet-base-v2')
    
    agent_a = AECP(LocalModelAdapter(model_a), agent_id="A")
    agent_b = AECP(LocalModelAdapter(model_b), agent_id="B")
    
    print("🤝 Calibrating...")
    agent_a.calibrate_with(agent_b)
    
    # Test Data
    queries = [
        "How to train a neural network?",
        "Best way to cook pasta",
        "The capital of Japan",
        "Quantum entanglement explained",
        "Signs of a recession"
    ]
    
    # For each query, we have a positive match and some distractors
    test_cases = [
        {
            "query": "How to train a neural network?",
            "positive": "Guidelines for stochastic gradient descent in deep learning",
            "negatives": ["Pasta recipes", "Tokyo city guide", "Stock market analysis", "How to fix a leaky faucet"]
        },
        {
            "query": "Best way to cook pasta",
            "positive": "Al dente boiling times for different noodle types",
            "negatives": ["Backpropagation in CNNs", "Kyoto travel tips", "Inflation rates 2024", "Plumbing basics"]
        }
    ]
    
    print("\n🔍 Testing Retrieval Preservation...")
    print(f"{'Query':<30} | {'Method':<10} | {'Rank of Positive':<20} | {'Success':<10}")
    print("-" * 80)
    
    for case in test_cases:
        query = case["query"]
        positive = case["positive"]
        candidates = [positive] + case["negatives"]
        
        # Ground Truth: Everything in Model B
        truth_query_b = agent_b.embed(query)
        candidates_b = [agent_b.embed(c) for c in candidates]
        
        # Ground Truth Similarities
        truth_sims = [aecp_sim(truth_query_b, c) for c in candidates_b]
        truth_rank = np.argsort(truth_sims)[::-1].tolist().index(0) + 1
        
        # AECP Transfer: Query from A, candidates in B
        query_a = agent_a.embed(query)
        transferred_query_b = agent_a.transfer_to("B", query).embedding
        
        aecp_sims = [aecp_sim(transferred_query_b, c) for c in candidates_b]
        aecp_rank = np.argsort(aecp_sims)[::-1].tolist().index(0) + 1
        
        print(f"{query[:30]:<30} | {'GroundTruth':<10} | {truth_rank:<20} | {'YES' if truth_rank==1 else 'NO'}")
        print(f"{'':<30} | {'AECP':<10} | {aecp_rank:<20} | {'YES' if aecp_rank==1 else 'NO'}")
        print("-" * 80)

if __name__ == "__main__":
    test_retrieval_preservation()
