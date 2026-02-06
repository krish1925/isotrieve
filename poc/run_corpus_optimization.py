"""
Quick runner for corpus size optimization experiment.

This focuses on finding the optimal corpus size without running
non-linear experiments (which take longer).
"""

import os
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import warnings
warnings.filterwarnings('ignore')

from optimization_experiments import (
    experiment_corpus_size_sweep,
    analyze_optimal_corpus_size,
    plot_results
)
from sentence_transformers import SentenceTransformer
import json
from datetime import datetime

def main():
    print("="*70)
    print(" CORPUS SIZE OPTIMIZATION EXPERIMENT")
    print("="*70)
    print("\nGoal: Find optimal corpus size where marginal improvement")
    print("      per minute of training time is < 1%")
    print()
    
    # Load embedding models
    print("Loading embedding models...")
    embedder1 = SentenceTransformer('all-MiniLM-L6-v2')  # 384d
    embedder2 = SentenceTransformer('all-mpnet-base-v2')  # 768d
    print("✓ Models loaded\n")
    
    # Test smaller range for faster experiments - will terminate early if no improvement
    corpus_sizes = [1000, 2000, 5000, 10000, 20000]
    
    print(f"Testing corpus sizes: {corpus_sizes}")
    print("(This may take 30-60 minutes depending on sizes tested)\n")
    
    corpus_results = experiment_corpus_size_sweep(
        embedder1, embedder2,
        corpus_sizes=corpus_sizes,
        val_size=2000,  # Reduced for speed
        test_size=500,  # Reduced for speed
        early_termination=True,
        no_improvement_threshold=3,
        timeout_seconds=180  # 3 minutes per test
    )
    
    # Analyze optimal corpus size
    optimal_analysis = analyze_optimal_corpus_size(corpus_results)
    
    print(f"\n{'='*70}")
    print("OPTIMAL CORPUS SIZE ANALYSIS")
    print(f"{'='*70}")
    print(f"\nOptimal corpus size: {optimal_analysis['optimal_corpus_size']:,}")
    print(f"  Validation Similarity: {optimal_analysis['optimal_similarity']:.4f}")
    print(f"  Training Time: {optimal_analysis['optimal_time']/60:.2f} minutes")
    
    # Show marginal improvements
    print(f"\nMarginal Improvements:")
    sizes = optimal_analysis['all_sizes']
    improvements = optimal_analysis['marginal_improvements']
    time_increases = optimal_analysis['time_increases']
    
    for i in range(len(improvements)):
        size_from = sizes[i]
        size_to = sizes[i+1]
        improvement = improvements[i]
        time_inc = time_increases[i]
        
        if time_inc > 0:
            improvement_per_min = improvement / (time_inc / 60)
            print(f"  {size_from:,} → {size_to:,}: "
                  f"+{improvement:.4f} similarity, "
                  f"+{time_inc/60:.2f} min, "
                  f"{improvement_per_min:.4f} sim/min")
        else:
            print(f"  {size_from:,} → {size_to:,}: "
                  f"+{improvement:.4f} similarity, "
                  f"+{time_inc:.2f} sec")
    
    # Save results
    import os
    os.makedirs("reports", exist_ok=True)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "corpus_size_experiment": corpus_results,
        "optimal_analysis": optimal_analysis
    }
    
    with open("reports/corpus_optimization_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved results: reports/corpus_optimization_results.json")
    
    # Generate plots
    plot_results(corpus_results, {})
    
    # Print detailed results table
    print(f"\n{'='*70}")
    print("DETAILED RESULTS")
    print(f"{'='*70}")
    print(f"\n{'Size':>10} {'Time (min)':>12} {'Val Sim':>10} {'Test Sim':>10} {'Efficiency':>12}")
    print("-" * 70)
    
    for size in sorted(corpus_results.keys()):
        r = corpus_results[size]
        time_min = r['training_time_seconds'] / 60
        efficiency = r['validation_similarity'] / time_min if time_min > 0 else 0
        print(f"{size:>10,} {time_min:>12.2f} {r['validation_similarity']:>10.4f} "
              f"{r['test_similarity']:>10.4f} {efficiency:>12.4f}")
    
    print("\n✓ Experiment complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
