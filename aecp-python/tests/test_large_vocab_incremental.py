"""
Test large vocabulary calibration with incremental validation.

Tests calibration with up to 50K vocabulary, validating every 500 words.
"""

import pytest
import numpy as np
import json
import os
import time
from typing import List
from tqdm import tqdm
from datetime import datetime
from aecp.adapters import MockAdapter
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


def generate_graphs(results: dict, output_dir: str, timestamp: str):
    """Generate visualization graphs from incremental calibration results."""
    steps = results["steps"]
    vocab_sizes = results["vocab_sizes"]
    val_sims = results["validation_similarities"]
    train_sims = results["training_similarities"]
    step_times = results["step_times"]
    cumulative_times = results["cumulative_times"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Vocabulary size vs Validation similarity
    axes[0, 0].plot(vocab_sizes, val_sims, 'o-', linewidth=2, markersize=4, color='blue')
    axes[0, 0].set_xlabel('Vocabulary Size', fontsize=11)
    axes[0, 0].set_ylabel('Validation Similarity', fontsize=11)
    axes[0, 0].set_title('Validation Similarity vs Vocabulary Size', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xscale('log')
    
    # Plot 2: Step time vs Vocabulary size (shows why time increases)
    axes[0, 1].plot(vocab_sizes, step_times, 'o-', linewidth=2, markersize=4, color='orange')
    axes[0, 1].set_xlabel('Vocabulary Size', fontsize=11)
    axes[0, 1].set_ylabel('Step Time (seconds)', fontsize=11)
    axes[0, 1].set_title('Step Time vs Vocabulary Size\n(Time increases because more words encoded)', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_yscale('log')
    
    # Plot 3: Cumulative time
    axes[1, 0].plot(vocab_sizes, cumulative_times, 'o-', linewidth=2, markersize=4, color='green')
    axes[1, 0].set_xlabel('Vocabulary Size', fontsize=11)
    axes[1, 0].set_ylabel('Cumulative Time (seconds)', fontsize=11)
    axes[1, 0].set_title('Cumulative Training Time', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale('log')
    
    # Plot 4: Training vs Validation similarity
    axes[1, 1].plot(vocab_sizes, train_sims, 'o-', linewidth=2, markersize=4, label='Training', color='purple')
    axes[1, 1].plot(vocab_sizes, val_sims, 'o-', linewidth=2, markersize=4, label='Validation', color='blue')
    axes[1, 1].set_xlabel('Vocabulary Size', fontsize=11)
    axes[1, 1].set_ylabel('Similarity', fontsize=11)
    axes[1, 1].set_title('Training vs Validation Similarity', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xscale('log')
    
    plt.tight_layout()
    plot_path = f"{output_dir}/incremental_calibration_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved graph to: {plot_path}")


def generate_large_vocabulary(size: int = 50000, seed: int = 42) -> List[str]:
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
        "network", "server", "client", "endpoint", "gateway", "application",
        "software", "hardware", "platform", "environment", "infrastructure",
        "security", "authentication", "authorization", "encryption", "token",
        "efficient", "effective", "robust", "reliable", "scalable", "maintainable",
        "secure", "stable", "consistent", "comprehensive", "accurate", "precise",
        "fast", "quick", "rapid", "slow", "gradual", "incremental",
        "simple", "complex", "sophisticated", "advanced", "basic", "fundamental",
        "modern", "contemporary", "current", "recent", "new", "innovative",
        "machine", "learning", "artificial", "intelligence", "neural", "deep",
        "data", "science", "analytics", "statistics", "probability", "inference",
        "training", "testing", "validation", "performance", "accuracy", "precision",
        "optimization", "gradient", "descent", "backpropagation", "tensor", "matrix",
    ]
    
    # Generate variations
    for i in range(size):
        if i < len(base_words):
            vocabulary.append(base_words[i])
        else:
            # Generate variations
            base_idx = i % len(base_words)
            base = base_words[base_idx]
            
            # Add variations
            variations = [
                base,
                f"{base}_{i}",
                f"{base}s",
                f"{base}ing",
                f"{base}ed",
                f"the_{base}",
                f"{base}_system",
                f"{base}_model",
                f"{base}_algorithm",
                f"advanced_{base}",
                f"modern_{base}",
                f"efficient_{base}",
                f"{base}_implementation",
                f"{base}_framework",
                f"{base}_architecture",
            ]
            
            vocab_idx = i // len(base_words)
            if vocab_idx < len(variations):
                vocabulary.append(variations[vocab_idx])
            else:
                vocabulary.append(f"term_{i}")
    
    # Shuffle to ensure random distribution
    random.shuffle(vocabulary)
    
    return vocabulary[:size]


def calibrate_incremental(
    agent_a: AECP,
    agent_b: AECP,
    vocabulary: List[str],
    validation_vocab: List[str],
    step_size: int = 500,
    verbose: bool = True
) -> dict:
    """
    Calibrate incrementally, validating every step_size words.
    
    Args:
        agent_a: First agent
        agent_b: Second agent
        vocabulary: Training vocabulary
        validation_vocab: Validation vocabulary
        step_size: Validate every N words
        verbose: Print progress
        
    Returns:
        Dictionary with incremental results
    """
    results = {
        "steps": [],
        "vocab_sizes": [],
        "validation_similarities": [],
        "training_similarities": [],
        "step_times": [],  # Track time per step
        "cumulative_times": [],  # Track cumulative time
    }
    
    # Encode validation vocabulary once (for efficiency)
    if verbose:
        print(f"\nEncoding validation vocabulary ({len(validation_vocab)} items)...")
    
    # Handle both MockAdapter and real embedders
    if hasattr(agent_a.embedder, 'embed_batch'):
        val_emb_a = np.array(agent_a.embedder.embed_batch(validation_vocab))
        val_emb_b = np.array(agent_b.embedder.embed_batch(validation_vocab))
    else:
        # Real embedding models (HuggingFaceAdapter has embed_batch)
        val_emb_a = np.array(agent_a.embedder.embed_batch(validation_vocab))
        val_emb_b = np.array(agent_b.embedder.embed_batch(validation_vocab))
    
    # Calculate total steps
    total_steps = len(range(step_size, len(vocabulary) + 1, step_size))
    
    # Incremental calibration with progress bar
    steps_range = range(step_size, len(vocabulary) + 1, step_size)
    progress_bar = tqdm(steps_range, desc="Calibrating", unit="step", ncols=100) if verbose else steps_range
    
    cumulative_time = 0.0
    for step in progress_bar:
        current_vocab = vocabulary[:step]
        
        # Update progress bar description
        if verbose:
            progress_bar.set_description(f"Step {step:,}/{len(vocabulary):,}")
        
        # Measure step time
        step_start_time = time.time()
        
        # Encode current vocabulary
        # NOTE: Step time increases because we encode MORE vocabulary each step:
        # Step 1: 500 words, Step 2: 1000 words, Step 3: 1500 words, etc.
        # This is expected - each step processes more data
        if hasattr(agent_a.embedder, 'embed_batch'):
            train_emb_a = np.array(agent_a.embedder.embed_batch(current_vocab))
            train_emb_b = np.array(agent_b.embedder.embed_batch(current_vocab))
        else:
            # Real embedding models (HuggingFaceAdapter, etc.) - use batch encoding
            if hasattr(agent_a.embedder, 'embed_batch'):
                train_emb_a = np.array(agent_a.embedder.embed_batch(current_vocab))
                train_emb_b = np.array(agent_b.embedder.embed_batch(current_vocab))
            else:
                # Fallback: individual embedding
                train_emb_a_list = [agent_a.embedder.embed(text) for text in current_vocab]
                train_emb_b_list = [agent_b.embedder.embed(text) for text in current_vocab]
                train_emb_a = np.array(train_emb_a_list)
                train_emb_b = np.array(train_emb_b_list)
        
        # Compute transfer matrices
        W_AB, W_BA = compute_transfer_matrices(train_emb_a, train_emb_b)
        
        # Evaluate on training data (sample for speed)
        train_sample_size = min(1000, len(train_emb_a))
        train_sample_a = train_emb_a[:train_sample_size]
        train_sample_b = train_emb_b[:train_sample_size]
        
        train_metrics = evaluate_transfer_quality(
            train_sample_a, train_sample_b, W_AB, W_BA
        )
        # Use forward similarity, not round-trip! Round-trip is misleading with
        # underdetermined systems (e.g., 500 samples < 768 dims):
        # - Forward: 384->768 is underdetermined (poor quality)
        # - Backward: 768->384 is overdetermined (perfect reconstruction)
        # - Round-trip = 1.0 even though forward transfer is garbage!
        # Forward similarity measures actual transfer quality.
        train_sim = train_metrics["forward_mean_similarity"]
        
        # Evaluate on validation set
        val_metrics = evaluate_transfer_quality(
            val_emb_a, val_emb_b, W_AB, W_BA
        )
        val_sim = val_metrics["forward_mean_similarity"]
        
        # Calculate step time
        step_time = time.time() - step_start_time
        cumulative_time += step_time
        
        # Store results
        results["steps"].append(step)
        results["vocab_sizes"].append(len(current_vocab))
        results["validation_similarities"].append(val_sim)
        results["training_similarities"].append(train_sim)
        results["step_times"].append(step_time)
        results["cumulative_times"].append(cumulative_time)
        
        # Update progress bar with current metrics
        if verbose:
            prev_val_sim = results['validation_similarities'][-2] if len(results['validation_similarities']) > 1 else 0.0
            improvement = val_sim - prev_val_sim
            progress_bar.set_postfix({
                'train_sim': f'{train_sim:.4f}',
                'val_sim': f'{val_sim:.4f}',
                'improvement': f'{improvement:+.4f}',
                'step_time': f'{step_time:.1f}s'
            })
    
    return results


@pytest.fixture
def large_vocab_agents():
    """Create agents for large vocabulary testing with REAL embedding models."""
    # Use HuggingFaceAdapter which wraps SentenceTransformer properly
    import os
    os.environ['USE_TF'] = '0'
    os.environ['USE_TORCH'] = '1'
    
    from aecp.adapters.huggingface import HuggingFaceAdapter
    
    # Use real embedding models for actual calibration
    embedder_a = HuggingFaceAdapter(model='all-MiniLM-L6-v2')  # 384d
    embedder_b = HuggingFaceAdapter(model='all-mpnet-base-v2')  # 768d
    
    agent_a = AECP(embedder_a, agent_id="agent_a")
    agent_b = AECP(embedder_b, agent_id="agent_b")
    return agent_a, agent_b


def test_incremental_calibration_50k(large_vocab_agents):
    """Test incremental calibration with 50K vocabulary, validating every 500 words."""
    agent_a, agent_b = large_vocab_agents
    
    # Generate large vocabulary
    print("\nGenerating 50K vocabulary...")
    full_vocab = generate_large_vocabulary(50000)
    
    # Shuffle again before splitting to ensure random distribution
    import random
    random.shuffle(full_vocab)
    
    # Split into train and validation (90/10)
    split_idx = int(len(full_vocab) * 0.9)
    train_vocab = full_vocab[:split_idx]  # 45K
    val_vocab = full_vocab[split_idx:]    # 5K
    
    print(f"✓ Shuffled vocabulary before splitting (ensures train/val have similar distributions)")
    
    print(f"Training vocabulary: {len(train_vocab):,} items")
    print(f"Validation vocabulary: {len(val_vocab):,} items")
    
    # Run incremental calibration
    results = calibrate_incremental(
        agent_a, agent_b,
        vocabulary=train_vocab,
        validation_vocab=val_vocab,
        step_size=500,
        verbose=True
    )
    
    # Assertions
    assert len(results["steps"]) > 0
    assert len(results["validation_similarities"]) == len(results["steps"])
    
    # Check that we tested multiple steps
    assert len(results["steps"]) >= 10, f"Expected at least 10 steps, got {len(results['steps'])}"
    
    # Check that validation similarity improves or stabilizes
    final_val_sim = results["validation_similarities"][-1]
    assert final_val_sim > 0.5, f"Final validation similarity too low: {final_val_sim:.4f}"
    
    # Check that similarity generally improves (or at least doesn't degrade significantly)
    if len(results["validation_similarities"]) > 1:
        first_sim = results["validation_similarities"][0]
        last_sim = results["validation_similarities"][-1]
        # Allow some variance but shouldn't degrade significantly
        assert last_sim >= first_sim - 0.1, \
            f"Similarity degraded too much: {first_sim:.4f} -> {last_sim:.4f}"
    
    print(f"\n{'='*60}")
    print("INCREMENTAL CALIBRATION RESULTS")
    print(f"{'='*60}")
    print(f"Total steps: {len(results['steps'])}")
    print(f"Final validation similarity: {final_val_sim:.4f}")
    print(f"Best validation similarity: {max(results['validation_similarities']):.4f}")
    print(f"Final training similarity: {results['training_similarities'][-1]:.4f}")
    
    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")
    print(f"{'Step':>10} {'Vocab Size':>12} {'Train Sim':>12} {'Val Sim':>12} {'Improvement':>12}")
    print("-" * 70)
    prev_val_sim = 0.0
    for i, step in enumerate(results["steps"]):
        val_sim = results['validation_similarities'][i]
        improvement = val_sim - prev_val_sim if i > 0 else val_sim
        print(f"{step:>10,} {results['vocab_sizes'][i]:>12,} "
              f"{results['training_similarities'][i]:>12.4f} "
              f"{val_sim:>12.4f} {improvement:>+12.4f}")
        prev_val_sim = val_sim
    
    # Save results to JSON
    output_dir = "test_results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = f"{output_dir}/incremental_calibration_{timestamp}.json"
    
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved results to: {json_path}")
    
    # Generate graphs
    if PLOTTING_AVAILABLE:
        generate_graphs(results, output_dir, timestamp)
    else:
        print("⚠️  Matplotlib not available, skipping graph generation")


def test_incremental_calibration_smaller(large_vocab_agents):
    """Test incremental calibration with smaller vocabulary for faster testing."""
    agent_a, agent_b = large_vocab_agents
    
    # Generate smaller vocabulary for faster test
    print("\nGenerating 5K vocabulary (for faster test)...")
    full_vocab = generate_large_vocabulary(5000)
    
    # Shuffle again before splitting to ensure random distribution
    import random
    random.shuffle(full_vocab)
    
    # Split into train and validation (90/10)
    split_idx = int(len(full_vocab) * 0.9)
    train_vocab = full_vocab[:split_idx]  # 4.5K
    val_vocab = full_vocab[split_idx:]    # 500
    
    print(f"✓ Shuffled vocabulary before splitting (ensures train/val have similar distributions)")
    
    print(f"Training vocabulary: {len(train_vocab):,} items")
    print(f"Validation vocabulary: {len(val_vocab):,} items")
    
    # Run incremental calibration with smaller steps
    results = calibrate_incremental(
        agent_a, agent_b,
        vocabulary=train_vocab,
        validation_vocab=val_vocab,
        step_size=500,  # Validate every 500 words
        verbose=True
    )
    
    # Assertions
    assert len(results["steps"]) > 0
    assert len(results["validation_similarities"]) == len(results["steps"])
    
    # Check that validation similarity is reasonable
    final_val_sim = results["validation_similarities"][-1]
    assert final_val_sim > 0.5, f"Final validation similarity too low: {final_val_sim:.4f}"
    
    # Save results to JSON
    output_dir = "test_results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = f"{output_dir}/incremental_calibration_smaller_{timestamp}.json"
    
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved results to: {json_path}")
    
    # Generate graphs
    if PLOTTING_AVAILABLE:
        generate_graphs(results, output_dir, f"smaller_{timestamp}")
    else:
        print("⚠️  Matplotlib not available, skipping graph generation")
