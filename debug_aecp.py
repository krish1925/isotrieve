
import numpy as np
from sentence_transformers import SentenceTransformer
from aecp import AECP
from aecp.adapters import LocalModelAdapter
from aecp.matrix import cosine_similarity as aecp_sim

def debug_aecp():
    print("Loading models...")
    model_a = SentenceTransformer('all-MiniLM-L6-v2')
    model_b = SentenceTransformer('all-mpnet-base-v2')
    
    adapter_a = LocalModelAdapter(model_a)
    adapter_b = LocalModelAdapter(model_b)
    
    print(f"Model A dimensions: {adapter_a.get_dimensions()}")
    print(f"Model B dimensions: {adapter_b.get_dimensions()}")
    
    agent_a = AECP(adapter_a, agent_id="A")
    agent_b = AECP(adapter_b, agent_id="B")
    
    print("Calibrating A -> B...")
    # Use a very small vocabulary for quick debug if needed, 
    # but let's stick to default to see what happened.
    res = agent_a.calibrate_with(agent_b)
    print(f"Calibration Validation Sim: {res.validation_similarity:.4f}")
    
    phrase = "Artificial intelligence is transforming the world"
    print(f"\nTesting phrase: '{phrase}'")
    
    # Ground truth
    emb_a = agent_a.embed(phrase)
    emb_b = agent_b.embed(phrase)
    
    print(f"Emb A shape: {emb_a.shape}, Norm: {np.linalg.norm(emb_a):.4f}")
    print(f"Emb B shape: {emb_b.shape}, Norm: {np.linalg.norm(emb_b):.4f}")
    
    # Manual transfer
    matrix_AB = agent_a.transfer_matrices["A_B"].matrix_AB
    print(f"Matrix AB shape: {matrix_AB.shape}")
    
    transferred_b = emb_a @ matrix_AB
    print(f"Transferred B shape: {transferred_b.shape}, Norm: {np.linalg.norm(transferred_b):.4f}")
    
    # Manual Similarity
    sim = aecp_sim(emb_b, transferred_b)
    print(f"Manual Cosine Sim (B vs Transferred B): {sim:.4f}")
    
    # Protocol transfer
    transfer_obj = agent_a.transfer_to("B", phrase)
    sim_protocol = aecp_sim(emb_b, transfer_obj.embedding)
    print(f"Protocol Cosine Sim: {sim_protocol:.4f}")
    
    # Check what calibration reported for this specific pair
    from aecp.matrix import evaluate_transfer_quality
    metrics = evaluate_transfer_quality(
        np.array([emb_a]), 
        np.array([emb_b]), 
        matrix_AB
    )
    print(f"Library evaluation for this single pair: {metrics['forward_mean_similarity']:.4f}")

if __name__ == "__main__":
    debug_aecp()
