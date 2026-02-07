"""
Compare different transfer methods:
1. Linear transfer matrices (current approach)
2. Direct text-based round-trip (e1 -> text -> e2 -> text -> e1')
3. Manifold approach (PCA + linear, simple autoencoder)
"""

import pytest
import numpy as np
import json
import os
import time
from typing import List, Tuple, Dict
from tqdm import tqdm
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
# HuggingFaceAdapter imported in fixture
from aecp import AECP
from aecp.matrix import compute_transfer_matrices, evaluate_transfer_quality, cosine_similarity

# Try to import matplotlib for plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


def find_nearest_text(embedding: np.ndarray, vocab_embeddings: np.ndarray, vocabulary: List[str]) -> str:
    """Find the nearest text in vocabulary by cosine similarity."""
    # Normalize embeddings for cosine similarity
    embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
    vocab_norms = vocab_embeddings / (np.linalg.norm(vocab_embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Compute cosine similarities
    similarities = vocab_norms @ embedding_norm
    
    # Find index of maximum similarity
    nearest_idx = np.argmax(similarities)
    return vocabulary[nearest_idx]


def direct_text_roundtrip(
    emb_a: np.ndarray,
    vocab_a: np.ndarray,
    vocab_b: np.ndarray,
    vocabulary: List[str],
    agent_b: AECP,
    agent_a: AECP
) -> Tuple[np.ndarray, float]:
    """
    Direct text-based round-trip: e1 -> text -> e2 -> text -> e1'
    
    Args:
        emb_a: Original embedding from agent A
        vocab_a: Vocabulary embeddings from agent A
        vocab_b: Vocabulary embeddings from agent B
        vocabulary: List of vocabulary texts
        agent_b: Agent B
        agent_a: Agent A
        
    Returns:
        Tuple of (reconstructed embedding, similarity)
    """
    # Step 1: e1 -> find nearest text
    nearest_text = find_nearest_text(emb_a, vocab_a, vocabulary)
    
    # Step 2: text -> encode with agent B -> e2
    emb_b = np.array(agent_b.embedder.embed(nearest_text))
    
    # Step 3: e2 -> find nearest text (in agent B's space)
    nearest_text_b = find_nearest_text(emb_b, vocab_b, vocabulary)
    
    # Step 4: text -> encode with agent A -> e1'
    emb_a_reconstructed = np.array(agent_a.embedder.embed(nearest_text_b))
    
    # Step 5: Compare e1 and e1'
    sim = cosine_similarity(emb_a, emb_a_reconstructed)
    
    return emb_a_reconstructed, sim


def manifold_transfer_pca(
    train_emb_a: np.ndarray,
    train_emb_b: np.ndarray,
    test_emb_a: np.ndarray,
    n_components: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Manifold approach using PCA + linear transfer.
    
    Args:
        train_emb_a: Training embeddings from agent A
        train_emb_b: Training embeddings from agent B
        test_emb_a: Test embeddings from agent A
        n_components: Number of PCA components
        
    Returns:
        Tuple of (transferred embeddings, transfer matrices)
    """
    # Reduce dimensionality with PCA
    n_components = min(n_components, train_emb_a.shape[1], train_emb_b.shape[1])
    
    pca_a = PCA(n_components=n_components)
    pca_b = PCA(n_components=n_components)
    
    train_emb_a_pca = pca_a.fit_transform(train_emb_a)
    train_emb_b_pca = pca_b.fit_transform(train_emb_b)
    
    # Compute linear transfer in PCA space
    W_AB_pca, W_BA_pca = compute_transfer_matrices(train_emb_a_pca, train_emb_b_pca)
    
    # Transfer test embeddings
    test_emb_a_pca = pca_a.transform(test_emb_a)
    test_emb_b_pca = test_emb_a_pca @ W_AB_pca
    test_emb_b = pca_b.inverse_transform(test_emb_b_pca)
    
    return test_emb_b, (W_AB_pca, W_BA_pca, pca_a, pca_b)


def manifold_transfer_mlp(
    train_emb_a: np.ndarray,
    train_emb_b: np.ndarray,
    test_emb_a: np.ndarray,
    hidden_sizes: Tuple[int, ...] = (512, 256),
    max_iter: int = 200
) -> np.ndarray:
    """
    Manifold approach using MLP (simple autoencoder-like).
    
    Args:
        train_emb_a: Training embeddings from agent A
        train_emb_b: Training embeddings from agent B
        test_emb_a: Test embeddings from agent A
        hidden_sizes: Hidden layer sizes
        max_iter: Maximum iterations
        
    Returns:
        Transferred embeddings
    """
    # Simple MLP: A -> B
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_sizes,
        max_iter=max_iter,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    mlp.fit(train_emb_a, train_emb_b)
    test_emb_b = mlp.predict(test_emb_a)
    
    return test_emb_b


def compare_transfer_methods(
    agent_a: AECP,
    agent_b: AECP,
    vocabulary: List[str],
    validation_vocab: List[str],
    step_size: int = 500,
    verbose: bool = True
) -> dict:
    """
    Compare different transfer methods incrementally.
    
    Args:
        agent_a: First agent
        agent_b: Second agent
        vocabulary: Training vocabulary
        validation_vocab: Validation vocabulary
        step_size: Validate every N words
        verbose: Print progress
        
    Returns:
        Dictionary with comparison results
    """
    results = {
        "steps": [],
        "vocab_sizes": [],
        "linear_forward_sims": [],
        "linear_roundtrip_sims": [],
        "direct_text_sims": [],
        "manifold_pca_sims": [],
        "manifold_mlp_sims": [],
        "step_times": [],
        "cumulative_times": [],
    }
    
    # Encode validation vocabulary once
    if verbose:
        print(f"\nEncoding validation vocabulary ({len(validation_vocab)} items)...")
    
    val_emb_a = np.array(agent_a.embedder.embed_batch(validation_vocab))
    val_emb_b = np.array(agent_b.embedder.embed_batch(validation_vocab))
    
    # Calculate total steps
    total_steps = len(range(step_size, len(vocabulary) + 1, step_size))
    
    # Incremental calibration with progress bar
    steps_range = range(step_size, len(vocabulary) + 1, step_size)
    progress_bar = tqdm(steps_range, desc="Comparing methods", unit="step", ncols=120) if verbose else steps_range
    
    cumulative_time = 0.0
    for step in progress_bar:
        step_start_time = time.time()
        current_vocab = vocabulary[:step]
        
        if verbose:
            progress_bar.set_description(f"Step {step:,}/{len(vocabulary):,}")
        
        # Encode current vocabulary
        train_emb_a = np.array(agent_a.embedder.embed_batch(current_vocab))
        train_emb_b = np.array(agent_b.embedder.embed_batch(current_vocab))
        
        # Method 1: Linear transfer matrices
        W_AB, W_BA = compute_transfer_matrices(train_emb_a, train_emb_b)
        
        # Evaluate linear forward transfer
        linear_forward_metrics = evaluate_transfer_quality(
            val_emb_a, val_emb_b, W_AB, W_BA
        )
        linear_forward_sim = linear_forward_metrics["forward_mean_similarity"]
        linear_roundtrip_sim = linear_forward_metrics["roundtrip_mean_similarity"]
        
        # Method 2: Direct text-based round-trip
        # Sample validation set for speed (evaluate on subset)
        sample_size = min(100, len(validation_vocab))
        sample_indices = np.random.choice(len(validation_vocab), sample_size, replace=False)
        
        direct_text_sims = []
        for idx in sample_indices:
            _, sim = direct_text_roundtrip(
                val_emb_a[idx],
                train_emb_a,
                train_emb_b,
                current_vocab,
                agent_b,
                agent_a
            )
            direct_text_sims.append(sim)
        direct_text_sim = np.mean(direct_text_sims)
        
        # Method 3: Manifold PCA
        try:
            val_emb_b_pca, _ = manifold_transfer_pca(
                train_emb_a, train_emb_b, val_emb_a[:sample_size]
            )
            manifold_pca_sims = [
                cosine_similarity(val_emb_b[i], val_emb_b_pca[i])
                for i in range(len(val_emb_b_pca))
            ]
            manifold_pca_sim = np.mean(manifold_pca_sims)
        except Exception as e:
            if verbose:
                print(f"\nWarning: PCA manifold failed: {e}")
            manifold_pca_sim = 0.0
        
        # Method 4: Manifold MLP (skip for small vocab sizes to save time)
        if step >= 2000:  # Only run MLP for larger vocab sizes
            try:
                val_emb_b_mlp = manifold_transfer_mlp(
                    train_emb_a, train_emb_b, val_emb_a[:sample_size],
                    hidden_sizes=(min(256, train_emb_a.shape[1] // 2),),  # Single hidden layer
                    max_iter=50  # Further reduced for speed
                )
                manifold_mlp_sims = [
                    cosine_similarity(val_emb_b[i], val_emb_b_mlp[i])
                    for i in range(len(val_emb_b_mlp))
                ]
                manifold_mlp_sim = np.mean(manifold_mlp_sims)
            except Exception as e:
                if verbose:
                    print(f"\nWarning: MLP manifold failed: {e}")
                manifold_mlp_sim = 0.0
        else:
            manifold_mlp_sim = 0.0  # Skip for small vocab
        
        # Calculate step time
        step_time = time.time() - step_start_time
        cumulative_time += step_time
        
        # Store results
        results["steps"].append(step)
        results["vocab_sizes"].append(len(current_vocab))
        results["linear_forward_sims"].append(linear_forward_sim)
        results["linear_roundtrip_sims"].append(linear_roundtrip_sim)
        results["direct_text_sims"].append(direct_text_sim)
        results["manifold_pca_sims"].append(manifold_pca_sim)
        results["manifold_mlp_sims"].append(manifold_mlp_sim)
        results["step_times"].append(step_time)
        results["cumulative_times"].append(cumulative_time)
        
        # Update progress bar
        if verbose:
            progress_bar.set_postfix({
                'linear_fwd': f'{linear_forward_sim:.3f}',
                'direct_txt': f'{direct_text_sim:.3f}',
                'pca': f'{manifold_pca_sim:.3f}',
                'mlp': f'{manifold_mlp_sim:.3f}',
                'time': f'{step_time:.1f}s'
            })
    
    return results


def generate_comparison_graphs(results: dict, output_dir: str, timestamp: str):
    """Generate comparison graphs."""
    if not PLOTTING_AVAILABLE:
        print("Matplotlib not available, skipping graph generation")
        return
    
    vocab_sizes = results["vocab_sizes"]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: All methods comparison
    axes[0, 0].plot(vocab_sizes, results["linear_forward_sims"], 'o-', label='Linear Forward', linewidth=2, markersize=4)
    axes[0, 0].plot(vocab_sizes, results["linear_roundtrip_sims"], 's-', label='Linear Round-trip', linewidth=2, markersize=4)
    axes[0, 0].plot(vocab_sizes, results["direct_text_sims"], '^-', label='Direct Text', linewidth=2, markersize=4)
    axes[0, 0].plot(vocab_sizes, results["manifold_pca_sims"], 'd-', label='Manifold PCA', linewidth=2, markersize=4)
    axes[0, 0].plot(vocab_sizes, results["manifold_mlp_sims"], 'v-', label='Manifold MLP', linewidth=2, markersize=4)
    axes[0, 0].set_xlabel('Vocabulary Size', fontsize=11)
    axes[0, 0].set_ylabel('Similarity', fontsize=11)
    axes[0, 0].set_title('Transfer Method Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].legend(loc='best')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xscale('log')
    
    # Plot 2: Linear vs Direct Text
    axes[0, 1].plot(vocab_sizes, results["linear_forward_sims"], 'o-', label='Linear Forward', linewidth=2, markersize=4, color='blue')
    axes[0, 1].plot(vocab_sizes, results["direct_text_sims"], '^-', label='Direct Text', linewidth=2, markersize=4, color='red')
    axes[0, 1].set_xlabel('Vocabulary Size', fontsize=11)
    axes[0, 1].set_ylabel('Similarity', fontsize=11)
    axes[0, 1].set_title('Linear vs Direct Text Transfer', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xscale('log')
    
    # Plot 3: Manifold methods
    axes[1, 0].plot(vocab_sizes, results["linear_forward_sims"], 'o-', label='Linear (baseline)', linewidth=2, markersize=4, color='gray', alpha=0.5)
    axes[1, 0].plot(vocab_sizes, results["manifold_pca_sims"], 'd-', label='Manifold PCA', linewidth=2, markersize=4, color='green')
    axes[1, 0].plot(vocab_sizes, results["manifold_mlp_sims"], 'v-', label='Manifold MLP', linewidth=2, markersize=4, color='purple')
    axes[1, 0].set_xlabel('Vocabulary Size', fontsize=11)
    axes[1, 0].set_ylabel('Similarity', fontsize=11)
    axes[1, 0].set_title('Manifold Methods Comparison', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale('log')
    
    # Plot 4: Step time
    axes[1, 1].plot(vocab_sizes, results["step_times"], 'o-', linewidth=2, markersize=4, color='orange')
    axes[1, 1].set_xlabel('Vocabulary Size', fontsize=11)
    axes[1, 1].set_ylabel('Step Time (seconds)', fontsize=11)
    axes[1, 1].set_title('Computation Time per Step', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plot_path = f"{output_dir}/transfer_comparison_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved comparison graph to: {plot_path}")


def generate_large_vocabulary(size: int = 10000, seed: int = 42) -> List[str]:
    """Generate a large vocabulary for testing."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    
    vocabulary = []
    
    # Base words
    base_words = [
        "analyze", "compute", "process", "optimize", "implement", "design",
        "test", "validate", "measure", "evaluate", "compare", "synthesize",
        "integrate", "deploy", "monitor", "debug", "refactor", "document",
        "create", "build", "develop", "organize", "manage", "execute",
        "transform", "convert", "translate", "interpret", "explain", "demonstrate",
        "algorithm", "function", "method", "procedure", "system", "model",
        "framework", "architecture", "structure", "component", "module", "interface",
        "protocol", "database", "repository", "storage", "memory", "cache",
        "network", "server", "client", "request", "response", "message",
        "data", "information", "knowledge", "insight", "analysis", "result",
        "performance", "efficiency", "optimization", "scalability", "reliability", "security",
        "machine", "learning", "neural", "network", "deep", "artificial",
        "intelligence", "natural", "language", "processing", "computer", "vision",
        "statistics", "probability", "distribution", "regression", "classification", "clustering",
        "python", "javascript", "java", "c++", "rust", "go",
        "programming", "software", "engineering", "development", "coding", "implementation",
    ]
    
    # Generate variations
    variations = [
        lambda w: w,
        lambda w: w.upper(),
        lambda w: w.capitalize(),
        lambda w: f"{w}_v1",
        lambda w: f"{w}_v2",
        lambda w: f"{w}_new",
        lambda w: f"{w}_old",
        lambda w: f"new_{w}",
        lambda w: f"old_{w}",
        lambda w: f"{w}_data",
        lambda w: f"{w}_model",
        lambda w: f"{w}_system",
        lambda w: f"{w}_test",
        lambda w: f"{w}_example",
        lambda w: f"{w}_demo",
    ]
    
    for i in range(size):
        base = base_words[i % len(base_words)]
        variation = variations[i // len(base_words) % len(variations)]
        vocabulary.append(variation(base))
    
    # Shuffle
    random.shuffle(vocabulary)
    
    return vocabulary


@pytest.fixture
def comparison_agents():
    """Create agents for comparison."""
    from aecp.adapters.huggingface import HuggingFaceAdapter
    
    # Use a small model and mock for faster testing if possible, 
    # but keep these for the "comparison" test if needed.
    # However, let's at least ensure they are shared/cached if possible.
    agent_a = AECP(
        agent_id="agent_a",
        embedder=HuggingFaceAdapter(model='all-MiniLM-L6-v2')  # 384d
    )
    agent_b = AECP(
        agent_id="agent_b",
        embedder=HuggingFaceAdapter(model='all-mpnet-base-v2')  # 768d
    )
    return agent_a, agent_b


@pytest.mark.slow
def test_transfer_comparison_2k(comparison_agents):
    """Compare transfer methods with 2K vocabulary (reduced from 10K)."""
    agent_a, agent_b = comparison_agents
    
    # Generate smaller vocabulary for standard testing
    print("Generating 2K vocabulary...")
    full_vocab = generate_large_vocabulary(size=2000, seed=42)
    
    # Split train/val
    split_idx = int(len(full_vocab) * 0.9)
    train_vocab = full_vocab[:split_idx]
    val_vocab = full_vocab[split_idx:]
    
    print(f"Training vocabulary: {len(train_vocab):,} items")
    print(f"Validation vocabulary: {len(val_vocab):,} items")
    
    # Run comparison
    results = compare_transfer_methods(
        agent_a, agent_b,
        train_vocab, val_vocab,
        step_size=500,
        verbose=True
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    json_path = f"{output_dir}/transfer_comparison_10k_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved results to: {json_path}")
    
    # Generate graphs
    generate_comparison_graphs(results, output_dir, f"10k_{timestamp}")
    
    # Print summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"Linear Forward:     {results['linear_forward_sims'][-1]:.4f}")
    print(f"Linear Round-trip:  {results['linear_roundtrip_sims'][-1]:.4f}")
    print(f"Direct Text:        {results['direct_text_sims'][-1]:.4f}")
    print(f"Manifold PCA:       {results['manifold_pca_sims'][-1]:.4f}")
    print(f"Manifold MLP:       {results['manifold_mlp_sims'][-1]:.4f}")
    print("="*60)
