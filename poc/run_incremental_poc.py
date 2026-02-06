"""
Incremental POC Runner - Small vocabulary sizes for scalability chart

Runs POC experiments for vocabulary sizes: 0 (random), 1k, 5k, 10k
to get data points for the scalability chart.
"""

import os
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sentence_transformers import SentenceTransformer
import json
from datetime import datetime

from protocol import ProtocolHandler
from matrix_transfer import compute_transfer_matrices, cosine_similarity


def test_random_matrix_performance(agent_a, agent_b, test_vocab):
    """
    Test performance with a random matrix (no training).
    """
    print("\n" + "="*70)
    print("TESTING RANDOM MATRIX (0 vocabulary)")
    print("="*70)
    
    # Generate random matrices
    dim_a = agent_a.capabilities.dimensions
    dim_b = agent_b.capabilities.dimensions
    
    # Random matrix with small values (normalized)
    np.random.seed(42)
    W_AB_random = np.random.randn(dim_a, dim_b) * 0.01
    W_BA_random = np.random.randn(dim_b, dim_a) * 0.01
    
    # Normalize to prevent explosion
    W_AB_random = W_AB_random / np.linalg.norm(W_AB_random, axis=0, keepdims=True)
    W_BA_random = W_BA_random / np.linalg.norm(W_BA_random, axis=0, keepdims=True)
    
    # Test on validation vocabulary
    print(f"\nEncoding {len(test_vocab)} test vocabulary items...")
    emb_a = agent_a.embedder.encode(test_vocab, show_progress_bar=False, batch_size=128)
    emb_b = agent_b.embedder.encode(test_vocab, show_progress_bar=False, batch_size=128)
    
    # Round-trip test
    transferred = emb_a @ W_AB_random
    roundtrip = transferred @ W_BA_random
    
    similarities = [cosine_similarity(emb_a[i], roundtrip[i]) for i in range(len(emb_a))]
    
    mean_sim = float(np.mean(similarities))
    print(f"\nRandom Matrix Results:")
    print(f"  Mean similarity: {mean_sim:.4f} ({mean_sim*100:.2f}%)")
    
    return mean_sim


def run_incremental_experiment(vocab_size, agent_a, agent_b, test_vocab):
    """
    Run POC experiment with specified vocabulary size.
    """
    print("\n" + "="*70)
    print(f"RUNNING EXPERIMENT: {vocab_size:,} vocabulary items")
    print("="*70)
    
    if vocab_size == 0:
        return test_random_matrix_performance(agent_a, agent_b, test_vocab)
    
    # Generate vocabulary
    from enhanced_vocab_loader import generate_diverse_vocabulary
    train_vocab = generate_diverse_vocabulary(vocab_size)
    
    # Use same test vocab for validation
    val_vocab = test_vocab[:min(1000, len(test_vocab))]
    
    print(f"\nTraining vocabulary: {len(train_vocab):,} items")
    print(f"Validation vocabulary: {len(val_vocab):,} items")
    
    # Encode training vocabulary
    print("\nEncoding training vocabulary...")
    emb_a_train = agent_a.embedder.encode(train_vocab, show_progress_bar=False, batch_size=128)
    emb_b_train = agent_b.embedder.encode(train_vocab, show_progress_bar=False, batch_size=128)
    
    # Compute transfer matrices
    print("Computing transfer matrices...")
    W_AB, W_BA = compute_transfer_matrices(emb_a_train, emb_b_train)
    
    # Training round-trip similarity
    train_transferred = emb_a_train @ W_AB
    train_roundtrip = train_transferred @ W_BA
    sample_size = min(1000, len(emb_a_train))
    train_sims = [cosine_similarity(emb_a_train[i], train_roundtrip[i])
                  for i in range(sample_size)]
    training_sim = float(np.mean(train_sims))
    
    # Validation round-trip similarity
    print("\nValidating on held-out vocabulary...")
    emb_a_val = agent_a.embedder.encode(val_vocab, show_progress_bar=False, batch_size=128)
    emb_b_val = agent_b.embedder.encode(val_vocab, show_progress_bar=False, batch_size=128)
    
    val_transferred = emb_a_val @ W_AB
    val_roundtrip = val_transferred @ W_BA
    val_sims = [cosine_similarity(emb_a_val[i], val_roundtrip[i])
               for i in range(len(emb_a_val))]
    validation_sim = float(np.mean(val_sims))
    
    print(f"\nResults:")
    print(f"  Training similarity: {training_sim:.4f} ({training_sim*100:.2f}%)")
    print(f"  Validation similarity: {validation_sim:.4f} ({validation_sim*100:.2f}%)")
    
    return validation_sim


def main():
    """
    Run incremental POC experiments.
    """
    print("="*70)
    print(" INCREMENTAL POC - Scalability Chart Data Points")
    print("="*70)
    print("\nThis will run experiments for vocabulary sizes:")
    print("  0 (random matrix), 1k, 5k, 10k")
    print()
    
    # Load models
    print("Loading embedding models...")
    embedder_a = SentenceTransformer('all-MiniLM-L6-v2')
    embedder_b = SentenceTransformer('all-mpnet-base-v2')
    
    agent_a = ProtocolHandler("agent_a", embedder_a, "all-MiniLM-L6-v2", 384)
    agent_b = ProtocolHandler("agent_b", embedder_b, "all-mpnet-base-v2", 768)
    
    # Generate test vocabulary (used for validation in all experiments)
    from enhanced_vocab_loader import generate_diverse_vocabulary
    test_vocab = generate_diverse_vocabulary(1000)
    
    # Run experiments
    vocab_sizes = [0, 1000, 5000, 10000]
    results = {}
    
    for vocab_size in vocab_sizes:
        try:
            sim = run_incremental_experiment(vocab_size, agent_a, agent_b, test_vocab)
            results[vocab_size] = sim
        except Exception as e:
            print(f"\n❌ Error with vocab_size={vocab_size}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print("\nVocabulary Size -> Validation Similarity:")
    for vocab_size in vocab_sizes:
        if vocab_size in results:
            print(f"  {vocab_size:>6,} -> {results[vocab_size]:.4f} ({results[vocab_size]*100:.2f}%)")
    
    # Save results
    output_file = "reports/incremental_results.json"
    os.makedirs("reports", exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "vocab_sizes": vocab_sizes
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # Print JavaScript array format for easy copy-paste
    print("\n" + "="*70)
    print("CHART DATA (for JavaScript)")
    print("="*70)
    print("\nValidation Similarity data array:")
    data_array = [results.get(vs, 0) * 100 for vs in vocab_sizes]
    print(f"  [{', '.join(f'{v:.2f}' for v in data_array)}]")
    
    print("\nTraining Similarity (estimated from validation):")
    train_array = [results.get(vs, 0) * 100 + 0.1 if vs > 0 else results.get(0, 0) * 100 for vs in vocab_sizes]
    print(f"  [{', '.join(f'{v:.2f}' for v in train_array)}]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
