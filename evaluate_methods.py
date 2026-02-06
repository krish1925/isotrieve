
import numpy as np
from sentence_transformers import SentenceTransformer
from aecp import AECP
from aecp.adapters import LocalModelAdapter
from aecp.matrix import compute_transfer_matrices, evaluate_transfer_quality
from aecp.vocabulary import get_default_vocabulary

def evaluate_methods():
    print("🔬 Evaluating Matrix Computation Methods")
    print("-" * 40)
    
    model_a = SentenceTransformer('all-MiniLM-L6-v2')
    model_b = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    train_vocab, val_vocab = get_default_vocabulary(train_size=500)
    
    emb_a_train = model_a.encode(train_vocab)
    emb_b_train = model_b.encode(train_vocab)
    emb_a_val = model_a.encode(val_vocab)
    emb_b_val = model_b.encode(val_vocab)
    
    # Method 1: Lstsq
    W_AB_l, W_BA_l = compute_transfer_matrices(emb_a_train, emb_b_train, method="lstsq")
    res_l = evaluate_transfer_quality(emb_a_val, emb_b_val, W_AB_l, W_BA_l)
    
    # Method 2: Ridge
    W_AB_r, W_BA_r = compute_transfer_matrices(emb_a_train, emb_b_train, method="ridge")
    res_r = evaluate_transfer_quality(emb_a_val, emb_b_val, W_AB_r, W_BA_r)
    
    print(f"Lstsq Validation Fidelity: {res_l['roundtrip_mean_similarity']:.4%}")
    print(f"Ridge Validation Fidelity: {res_r['roundtrip_mean_similarity']:.4%}")
    
    # Small perturbations to show Ridge robustness
    print("\nRobustness Test (Adding Noise to Source)")
    noise = np.random.normal(0, 0.05, emb_a_val.shape)
    emb_a_val_noisy = emb_a_val + noise
    
    # Test Lstsq with noise
    transferred_l = emb_a_val_noisy @ W_AB_l
    sim_l = np.mean([1 - np.linalg.norm(transferred_l[i]/np.linalg.norm(transferred_l[i]) - emb_b_val[i]/np.linalg.norm(emb_b_val[i]))**2/2 for i in range(len(emb_b_val))])
    
    # Test Ridge with noise
    transferred_r = emb_a_val_noisy @ W_AB_r
    sim_r = np.mean([1 - np.linalg.norm(transferred_r[i]/np.linalg.norm(transferred_r[i]) - emb_b_val[i]/np.linalg.norm(emb_b_val[i]))**2/2 for i in range(len(emb_b_val))])
    
    print(f"Lstsq Noisy Fidelity: {sim_l:.4%}")
    print(f"Ridge Noisy Fidelity: {sim_r:.4%}")
    
    improvement = (sim_r - sim_l) / sim_l
    print(f"Ridge improvement under noise: {improvement:.2%}")

if __name__ == "__main__":
    evaluate_methods()
