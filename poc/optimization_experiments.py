"""
Optimization Experiments for Embedding Transfer

This script experiments with:
1. Optimal corpus size (training time vs semantic loss tradeoff)
2. Non-linear transformation methods (PCA, autoencoders, neural networks)
3. Smaller manifold conversions

Goal: Find the sweet spot where increasing corpus size doesn't provide significant benefit.
"""

import os
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from matrix_transfer import cosine_similarity, compute_transfer_matrices, evaluate_transfer_quality
from enhanced_vocab_loader import generate_diverse_vocabulary, create_train_val_test_split

# Try to import sklearn for PCA and other methods
try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. PCA experiments will be skipped.")

# Try to import torch for neural network transformations
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Neural network experiments will be skipped.")

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Plots will be skipped.")


class NonLinearTransfer:
    """Non-linear transformation methods for embedding transfer."""
    
    def __init__(self, method: str = "pca", **kwargs):
        """
        Initialize non-linear transfer method.
        
        Args:
            method: 'pca', 'autoencoder', 'mlp', or 'linear' (baseline)
            **kwargs: Method-specific parameters
        """
        self.method = method
        self.kwargs = kwargs
        self.model = None
        self.scaler_source = None
        self.scaler_target = None
        
    def fit(
        self,
        embeddings_source: np.ndarray,
        embeddings_target: np.ndarray
    ) -> Dict[str, float]:
        """
        Fit the transformation model.
        
        Returns:
            Dictionary with training metrics
        """
        if self.method == "linear":
            # Baseline: use existing linear method
            self.W_forward, self.W_backward = compute_transfer_matrices(
                embeddings_source, embeddings_target
            )
            return {"method": "linear"}
            
        elif self.method == "pca":
            if not SKLEARN_AVAILABLE:
                raise ImportError("sklearn required for PCA method")
            return self._fit_pca(embeddings_source, embeddings_target)
            
        elif self.method == "autoencoder":
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch required for autoencoder method")
            return self._fit_autoencoder(embeddings_source, embeddings_target)
            
        elif self.method == "mlp":
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch required for MLP method")
            return self._fit_mlp(embeddings_source, embeddings_target)
            
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _fit_pca(
        self,
        embeddings_source: np.ndarray,
        embeddings_target: np.ndarray
    ) -> Dict[str, float]:
        """Fit PCA-based transformation."""
        # Normalize embeddings
        self.scaler_source = StandardScaler()
        self.scaler_target = StandardScaler()
        
        emb_source_scaled = self.scaler_source.fit_transform(embeddings_source)
        emb_target_scaled = self.scaler_target.fit_transform(embeddings_target)
        
        # Reduce dimensionality using PCA
        n_components = self.kwargs.get('n_components', min(128, embeddings_source.shape[1]))
        
        pca_source = PCA(n_components=n_components)
        pca_target = PCA(n_components=n_components)
        
        emb_source_pca = pca_source.fit_transform(emb_source_scaled)
        emb_target_pca = pca_target.fit_transform(emb_target_scaled)
        
        # Learn linear transformation in PCA space
        self.W_forward, self.W_backward = compute_transfer_matrices(
            emb_source_pca, emb_target_pca
        )
        
        self.pca_source = pca_source
        self.pca_target = pca_target
        
        return {
            "method": "pca",
            "n_components": n_components,
            "explained_variance_source": float(np.sum(pca_source.explained_variance_ratio_)),
            "explained_variance_target": float(np.sum(pca_target.explained_variance_ratio_))
        }
    
    def _fit_autoencoder(
        self,
        embeddings_source: np.ndarray,
        embeddings_target: np.ndarray
    ) -> Dict[str, float]:
        """Fit autoencoder-based transformation."""
        dim_source = embeddings_source.shape[1]
        dim_target = embeddings_target.shape[1]
        hidden_dim = self.kwargs.get('hidden_dim', min(256, (dim_source + dim_target) // 2))
        
        # Create autoencoder: source -> hidden -> target
        class AutoencoderTransfer(nn.Module):
            def __init__(self, dim_in, dim_hidden, dim_out):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(dim_in, dim_hidden),
                    nn.ReLU(),
                    nn.Linear(dim_hidden, dim_hidden // 2)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(dim_hidden // 2, dim_hidden),
                    nn.ReLU(),
                    nn.Linear(dim_hidden, dim_out)
                )
            
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded
        
        self.model = AutoencoderTransfer(dim_source, hidden_dim, dim_target)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.kwargs.get('lr', 0.001))
        criterion = nn.MSELoss()
        
        # Convert to tensors
        X = torch.FloatTensor(embeddings_source)
        y = torch.FloatTensor(embeddings_target)
        
        # Training
        n_epochs = self.kwargs.get('n_epochs', 50)
        batch_size = self.kwargs.get('batch_size', 128)
        
        losses = []
        for epoch in range(n_epochs):
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
        
        return {
            "method": "autoencoder",
            "hidden_dim": hidden_dim,
            "final_loss": float(np.mean(losses[-100:])),
            "n_epochs": n_epochs
        }
    
    def _fit_mlp(
        self,
        embeddings_source: np.ndarray,
        embeddings_target: np.ndarray
    ) -> Dict[str, float]:
        """Fit MLP-based transformation."""
        dim_source = embeddings_source.shape[1]
        dim_target = embeddings_target.shape[1]
        hidden_dims = self.kwargs.get('hidden_dims', [512, 256])
        
        class MLPTransfer(nn.Module):
            def __init__(self, dim_in, hidden_dims, dim_out):
                super().__init__()
                layers = []
                prev_dim = dim_in
                for h_dim in hidden_dims:
                    layers.append(nn.Linear(prev_dim, h_dim))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(0.1))
                    prev_dim = h_dim
                layers.append(nn.Linear(prev_dim, dim_out))
                self.network = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.network(x)
        
        self.model = MLPTransfer(dim_source, hidden_dims, dim_target)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.kwargs.get('lr', 0.001))
        criterion = nn.MSELoss()
        
        X = torch.FloatTensor(embeddings_source)
        y = torch.FloatTensor(embeddings_target)
        
        n_epochs = self.kwargs.get('n_epochs', 50)
        batch_size = self.kwargs.get('batch_size', 128)
        
        losses = []
        for epoch in range(n_epochs):
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
        
        return {
            "method": "mlp",
            "hidden_dims": hidden_dims,
            "final_loss": float(np.mean(losses[-100:])),
            "n_epochs": n_epochs
        }
    
    def transform_forward(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform embeddings forward."""
        if self.method == "linear":
            return embeddings @ self.W_forward
            
        elif self.method == "pca":
            emb_scaled = self.scaler_source.transform(embeddings)
            emb_pca = self.pca_source.transform(emb_scaled)
            emb_transformed = emb_pca @ self.W_forward
            # Inverse PCA
            emb_inv_pca = self.pca_target.inverse_transform(emb_transformed)
            emb_inv_scaled = self.scaler_target.inverse_transform(emb_inv_pca)
            return emb_inv_scaled
            
        elif self.method in ["autoencoder", "mlp"]:
            self.model.eval()
            with torch.no_grad():
                X = torch.FloatTensor(embeddings)
                output = self.model(X)
                return output.numpy()
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def transform_backward(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform embeddings backward (round-trip)."""
        if self.method == "linear":
            return embeddings @ self.W_backward
            
        elif self.method == "pca":
            # For PCA, we need to learn reverse transformation
            # For simplicity, use linear in PCA space
            emb_scaled = self.scaler_target.transform(embeddings)
            emb_pca = self.pca_target.transform(emb_scaled)
            emb_transformed = emb_pca @ self.W_backward
            emb_inv_pca = self.pca_source.inverse_transform(emb_transformed)
            emb_inv_scaled = self.scaler_source.inverse_transform(emb_inv_pca)
            return emb_inv_scaled
            
        elif self.method in ["autoencoder", "mlp"]:
            # For non-linear methods, backward is harder
            # We'd need a separate model - for now, use linear approximation
            # This is a limitation: true backward would need a separate model
            if hasattr(self, 'W_backward'):
                return embeddings @ self.W_backward
            else:
                # Fallback: learn linear backward transformation
                # This is approximate but works for evaluation
                raise NotImplementedError(
                    "Backward transformation for non-linear methods requires "
                    "separate model training. Use linear method for round-trip."
                )
        else:
            raise ValueError(f"Unknown method: {self.method}")


def plot_incremental_results(corpus_results: Dict, output_dir: str = "reports"):
    """Plot incremental results after each test."""
    if not PLOTTING_AVAILABLE:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    sizes = sorted(corpus_results.keys())
    similarities = [corpus_results[s]["validation_similarity"] for s in sizes]
    times = [corpus_results[s]["training_time_seconds"] / 60 for s in sizes]
    
    # Calculate efficiency (similarity per minute)
    efficiency = [sim / (time) if time > 0 else 0 for sim, time in zip(similarities, times)]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Size vs Similarity
    axes[0, 0].plot(sizes, similarities, 'o-', linewidth=2, markersize=8, color='blue')
    axes[0, 0].set_xlabel('Corpus Size', fontsize=11)
    axes[0, 0].set_ylabel('Validation Similarity', fontsize=11)
    axes[0, 0].set_title('Corpus Size vs Validation Similarity', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xscale('log')
    
    # Plot 2: Size vs Time
    axes[0, 1].plot(sizes, times, 'o-', linewidth=2, markersize=8, color='orange')
    axes[0, 1].set_xlabel('Corpus Size', fontsize=11)
    axes[0, 1].set_ylabel('Training Time (minutes)', fontsize=11)
    axes[0, 1].set_title('Corpus Size vs Training Time', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_yscale('log')
    
    # Plot 3: Efficiency (similarity per minute)
    axes[1, 0].plot(sizes, efficiency, 'o-', linewidth=2, markersize=8, color='green')
    axes[1, 0].set_xlabel('Corpus Size', fontsize=11)
    axes[1, 0].set_ylabel('Efficiency (Similarity/Minute)', fontsize=11)
    axes[1, 0].set_title('Training Efficiency', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale('log')
    
    # Plot 4: Marginal improvement
    if len(sizes) > 1:
        improvements = []
        for i in range(1, len(similarities)):
            improvements.append(similarities[i] - similarities[i-1])
        
        axes[1, 1].bar(range(len(improvements)), improvements, color='purple', alpha=0.7)
        axes[1, 1].set_xlabel('Size Step', fontsize=11)
        axes[1, 1].set_ylabel('Marginal Improvement', fontsize=11)
        axes[1, 1].set_title('Marginal Similarity Improvement', fontsize=12)
        axes[1, 1].set_xticks(range(len(improvements)))
        axes[1, 1].set_xticklabels([f"{sizes[i-1]}→{sizes[i]}" for i in range(1, len(sizes))], 
                                    rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/incremental_results.png", dpi=300, bbox_inches='tight')
    print(f"✓ Updated plot: {output_dir}/incremental_results.png")
    plt.close()


def check_early_termination(corpus_results: Dict, no_improvement_threshold: int = 3) -> bool:
    """
    Check if we should terminate early based on no efficiency gain.
    
    Returns True if 3-4 consecutive sizes show no efficiency improvement.
    """
    if len(corpus_results) < no_improvement_threshold + 1:
        return False
    
    sizes = sorted(corpus_results.keys())
    
    # Calculate efficiency for each size
    efficiencies = []
    for size in sizes:
        r = corpus_results[size]
        time_min = r['training_time_seconds'] / 60
        efficiency = r['validation_similarity'] / time_min if time_min > 0 else 0
        efficiencies.append(efficiency)
    
    # Check last N consecutive sizes for no improvement
    recent_efficiencies = efficiencies[-no_improvement_threshold:]
    
    # Check if all recent efficiencies are decreasing or flat
    is_decreasing = all(recent_efficiencies[i] <= recent_efficiencies[i-1] 
                       for i in range(1, len(recent_efficiencies)))
    
    # Also check if marginal similarity improvement is negligible
    recent_similarities = [corpus_results[s]["validation_similarity"] for s in sizes[-no_improvement_threshold:]]
    marginal_improvements = [recent_similarities[i] - recent_similarities[i-1] 
                            for i in range(1, len(recent_similarities))]
    
    # If improvements are all < 0.001 (0.1%), consider it no improvement
    negligible_improvement = all(abs(imp) < 0.001 for imp in marginal_improvements)
    
    return is_decreasing or negligible_improvement


def experiment_corpus_size_sweep(
    embedder1,
    embedder2,
    corpus_sizes: List[int],
    val_size: int = 5000,
    test_size: int = 1000,
    early_termination: bool = True,
    no_improvement_threshold: int = 3,
    timeout_seconds: int = 180  # 3 minutes timeout per test
) -> Dict:
    """
    Experiment with different corpus sizes to find optimal tradeoff.
    
    Args:
        embedder1: First embedding model
        embedder2: Second embedding model
        corpus_sizes: List of training corpus sizes to test
        val_size: Size of validation set
        test_size: Size of test set
        
    Returns:
        Dictionary with results for each corpus size
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: Corpus Size Optimization")
    print("="*70)
    
    # Generate vocabulary pool (limit to what we need)
    max_needed = max(corpus_sizes) + val_size + test_size
    print(f"\nGenerating vocabulary pool (max {max_needed:,} items)...")
    full_vocab = generate_diverse_vocabulary(max_needed)
    
    # Create fixed validation and test sets
    val_vocab = full_vocab[:val_size]
    test_vocab = full_vocab[val_size:val_size+test_size]
    
    results = {}
    
    for corpus_size in corpus_sizes:
        print(f"\n{'='*70}")
        print(f"Testing corpus size: {corpus_size:,}")
        print(f"{'='*70}")
        print(f"Timeout: {timeout_seconds}s per test")
        
        # Use different portion of vocabulary for each size
        train_vocab = full_vocab[val_size+test_size:val_size+test_size+corpus_size]
        
        # Measure training time
        start_time = time.time()
        
        try:
            # Encode vocabulary
            print(f"  Encoding {len(train_vocab):,} training items...")
            emb1_train = embedder1.encode(train_vocab, show_progress_bar=False, batch_size=128)
            
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                print(f"\n⚠️  TIMEOUT: Encoding took {elapsed:.1f}s (>{timeout_seconds}s). Skipping this size.")
                break
            
            emb2_train = embedder2.encode(train_vocab, show_progress_bar=False, batch_size=128)
            
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                print(f"\n⚠️  TIMEOUT: Encoding took {elapsed:.1f}s (>{timeout_seconds}s). Skipping this size.")
                break
            
            print(f"  Computing transfer matrices...")
            W_12, W_21 = compute_transfer_matrices(emb1_train, emb2_train)
            
            encoding_time = time.time() - start_time
            
            # Check timeout before evaluation
            if encoding_time > timeout_seconds:
                print(f"\n⚠️  TIMEOUT: Training took {encoding_time:.1f}s (>{timeout_seconds}s). Skipping this size.")
                break
            
            # Evaluate on validation set (use smaller sample if needed)
            val_sample_size = min(len(val_vocab), 2000)  # Limit validation size for speed
            val_vocab_sample = val_vocab[:val_sample_size]
            print(f"  Evaluating on validation set ({len(val_vocab_sample):,} items)...")
            emb1_val = embedder1.encode(val_vocab_sample, show_progress_bar=False, batch_size=128)
            emb2_val = embedder2.encode(val_vocab_sample, show_progress_bar=False, batch_size=128)
            
            val_metrics = evaluate_transfer_quality(emb1_val, emb2_val, W_12, W_21)
            
            # Evaluate on test set (use smaller sample)
            test_sample_size = min(len(test_vocab), 500)  # Limit test size for speed
            test_vocab_sample = test_vocab[:test_sample_size]
            print(f"  Evaluating on test set ({len(test_vocab_sample):,} items)...")
            emb1_test = embedder1.encode(test_vocab_sample, show_progress_bar=False, batch_size=128)
            emb2_test = embedder2.encode(test_vocab_sample, show_progress_bar=False, batch_size=128)
            
            test_metrics = evaluate_transfer_quality(emb1_test, emb2_test, W_12, W_21)
            
            total_time = time.time() - start_time
            
            # Final timeout check
            if total_time > timeout_seconds:
                print(f"\n⚠️  TIMEOUT: Total time {total_time:.1f}s exceeded {timeout_seconds}s. Skipping remaining sizes.")
                break
                
        except Exception as e:
            print(f"\n⚠️  ERROR during testing: {e}")
            print(f"  Skipping this corpus size and continuing...")
            continue
        
        results[corpus_size] = {
            "corpus_size": corpus_size,
            "training_time_seconds": total_time,
            "encoding_time_seconds": encoding_time,
            "validation_similarity": val_metrics["roundtrip_mean_similarity"],
            "test_similarity": test_metrics["roundtrip_mean_similarity"],
            "validation_std": val_metrics["roundtrip_std_similarity"],
            "test_std": test_metrics["roundtrip_std_similarity"],
            "validation_min": val_metrics["roundtrip_min_similarity"],
            "test_min": test_metrics["roundtrip_min_similarity"],
        }
        
        print(f"\n  Results:")
        print(f"    Training time: {total_time:.2f}s ({total_time/60:.2f} min)")
        print(f"    Validation similarity: {val_metrics['roundtrip_mean_similarity']:.4f}")
        print(f"    Test similarity: {test_metrics['roundtrip_mean_similarity']:.4f}")
        
        # Calculate efficiency
        efficiency = val_metrics['roundtrip_mean_similarity'] / (total_time / 60) if total_time > 0 else 0
        print(f"    Efficiency: {efficiency:.4f} similarity/min")
        
        # Plot incremental results
        plot_incremental_results(results)
        
        # Check for early termination
        if early_termination and check_early_termination(results, no_improvement_threshold):
            print(f"\n{'='*70}")
            print(f"EARLY TERMINATION: No efficiency gain for {no_improvement_threshold} consecutive sizes")
            print(f"{'='*70}")
            print(f"\nStopping at corpus size: {corpus_size:,}")
            print(f"Tested sizes: {sorted(results.keys())}")
            break
        
        # Breakpoint: show progress (informational, no pause)
        if corpus_size != corpus_sizes[-1]:
            print(f"\n{'='*70}")
            print(f"✓ Breakpoint: Completed {corpus_size:,}. Continuing to next size...")
            print(f"{'='*70}")
            print()  # Extra line for readability
    
    return results


def experiment_nonlinear_methods(
    embedder1,
    embedder2,
    train_vocab: List[str],
    val_vocab: List[str],
    test_vocab: List[str],
    methods: List[str] = None
) -> Dict:
    """
    Experiment with non-linear transformation methods.
    
    Args:
        embedder1: First embedding model
        embedder2: Second embedding model
        train_vocab: Training vocabulary
        val_vocab: Validation vocabulary
        test_vocab: Test vocabulary
        methods: List of methods to test ['linear', 'pca', 'autoencoder', 'mlp']
        
    Returns:
        Dictionary with results for each method
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: Non-Linear Transformation Methods")
    print("="*70)
    
    if methods is None:
        methods = ["linear"]
        if SKLEARN_AVAILABLE:
            methods.append("pca")
        if TORCH_AVAILABLE:
            methods.extend(["autoencoder", "mlp"])
    
    # Encode vocabulary
    print(f"\nEncoding vocabulary...")
    print(f"  Training: {len(train_vocab):,} items")
    emb1_train = embedder1.encode(train_vocab, show_progress_bar=True, batch_size=128)
    emb2_train = embedder2.encode(train_vocab, show_progress_bar=True, batch_size=128)
    
    print(f"  Validation: {len(val_vocab):,} items")
    emb1_val = embedder1.encode(val_vocab, show_progress_bar=True, batch_size=128)
    emb2_val = embedder2.encode(val_vocab, show_progress_bar=True, batch_size=128)
    
    print(f"  Test: {len(test_vocab):,} items")
    emb1_test = embedder1.encode(test_vocab, show_progress_bar=True, batch_size=128)
    emb2_test = embedder2.encode(test_vocab, show_progress_bar=True, batch_size=128)
    
    results = {}
    
    for method in methods:
        print(f"\n{'='*70}")
        print(f"Testing method: {method.upper()}")
        print(f"{'='*70}")
        
        # Method-specific parameters (reduced for speed)
        kwargs = {}
        if method == "pca":
            kwargs = {"n_components": 64}  # Reduced from 128
        elif method == "autoencoder":
            kwargs = {"hidden_dim": 128, "n_epochs": 20, "lr": 0.001}  # Reduced epochs
        elif method == "mlp":
            kwargs = {"hidden_dims": [256, 128], "n_epochs": 20, "lr": 0.001}  # Reduced epochs
        
        # Fit model
        start_time = time.time()
        transfer = NonLinearTransfer(method=method, **kwargs)
        fit_metrics = transfer.fit(emb1_train, emb2_train)
        
        # For non-linear methods, train backward model too
        if method in ["autoencoder", "mlp"]:
            print(f"  Training backward transformation...")
            transfer_backward = NonLinearTransfer(method=method, **kwargs)
            fit_metrics_backward = transfer_backward.fit(emb2_train, emb1_train)
            transfer.model_backward = transfer_backward.model
        
        training_time = time.time() - start_time
        
        # Evaluate on validation
        print(f"  Evaluating on validation set...")
        val_transferred = transfer.transform_forward(emb1_val)
        
        # For non-linear methods, use backward model
        if method in ["autoencoder", "mlp"] and hasattr(transfer, 'model_backward'):
            original_model = transfer.model
            transfer.model = transfer.model_backward
            val_roundtrip = transfer.transform_forward(val_transferred)
            transfer.model = original_model  # Restore forward model
        else:
            val_roundtrip = transfer.transform_backward(val_transferred)
        
        val_sims = [cosine_similarity(emb1_val[i], val_roundtrip[i])
                   for i in range(len(emb1_val))]
        
        val_metrics = {
            "mean": float(np.mean(val_sims)),
            "median": float(np.median(val_sims)),
            "std": float(np.std(val_sims)),
            "min": float(np.min(val_sims)),
            "max": float(np.max(val_sims))
        }
        
        # Evaluate on test
        print(f"  Evaluating on test set...")
        test_transferred = transfer.transform_forward(emb1_test)
        
        # For non-linear methods, use backward model
        if method in ["autoencoder", "mlp"] and hasattr(transfer, 'model_backward'):
            original_model = transfer.model
            transfer.model = transfer.model_backward
            test_roundtrip = transfer.transform_forward(test_transferred)
            transfer.model = original_model  # Restore forward model
        else:
            test_roundtrip = transfer.transform_backward(test_transferred)
        
        test_sims = [cosine_similarity(emb1_test[i], test_roundtrip[i])
                    for i in range(len(emb1_test))]
        
        test_metrics = {
            "mean": float(np.mean(test_sims)),
            "median": float(np.median(test_sims)),
            "std": float(np.std(test_sims)),
            "min": float(np.min(test_sims)),
            "max": float(np.max(test_sims))
        }
        
        results[method] = {
            "method": method,
            "training_time_seconds": training_time,
            "fit_metrics": fit_metrics,
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics
        }
        
        print(f"\n  Results:")
        print(f"    Training time: {training_time:.2f}s ({training_time/60:.2f} min)")
        print(f"    Validation similarity: {val_metrics['mean']:.4f}")
        print(f"    Test similarity: {test_metrics['mean']:.4f}")
    
    return results


def analyze_optimal_corpus_size(results: Dict) -> Dict:
    """
    Analyze results to find optimal corpus size.
    
    Finds the point where marginal improvement < threshold.
    """
    sizes = sorted(results.keys())
    similarities = [results[s]["validation_similarity"] for s in sizes]
    times = [results[s]["training_time_seconds"] for s in sizes]
    
    # Calculate marginal improvements
    improvements = []
    time_increases = []
    
    for i in range(1, len(sizes)):
        sim_improvement = similarities[i] - similarities[i-1]
        time_increase = times[i] - times[i-1]
        improvements.append(sim_improvement)
        time_increases.append(time_increase)
    
    # Find optimal: where improvement per minute < threshold
    threshold = 0.01  # 1% improvement per minute
    optimal_idx = len(sizes) - 1
    
    for i in range(len(improvements)):
        if time_increases[i] > 0:
            improvement_per_minute = improvements[i] / (time_increases[i] / 60)
            if improvement_per_minute < threshold:
                optimal_idx = i
                break
    
    optimal_size = sizes[optimal_idx]
    
    return {
        "optimal_corpus_size": optimal_size,
        "optimal_similarity": similarities[optimal_idx],
        "optimal_time": times[optimal_idx],
        "marginal_improvements": improvements,
        "time_increases": time_increases,
        "all_sizes": sizes,
        "all_similarities": similarities,
        "all_times": times
    }


def plot_results(corpus_results: Dict, nonlinear_results: Dict, output_dir: str = "reports"):
    """Generate visualization plots."""
    if not PLOTTING_AVAILABLE:
        print("Skipping plots (matplotlib not available)")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Corpus size vs similarity and time
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    sizes = sorted(corpus_results.keys())
    similarities = [corpus_results[s]["validation_similarity"] for s in sizes]
    times = [corpus_results[s]["training_time_seconds"] / 60 for s in sizes]  # Convert to minutes
    
    ax1.plot(sizes, similarities, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Corpus Size', fontsize=12)
    ax1.set_ylabel('Validation Similarity', fontsize=12)
    ax1.set_title('Corpus Size vs Semantic Similarity', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    ax2.plot(sizes, times, 'o-', color='orange', linewidth=2, markersize=8)
    ax2.set_xlabel('Corpus Size', fontsize=12)
    ax2.set_ylabel('Training Time (minutes)', fontsize=12)
    ax2.set_title('Corpus Size vs Training Time', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/corpus_size_optimization.png", dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved plot: {output_dir}/corpus_size_optimization.png")
    
    # Plot 2: Efficiency (similarity per minute)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    efficiency = [sim / (time / 60) for sim, time in zip(similarities, times)]
    ax.plot(sizes, efficiency, 'o-', color='green', linewidth=2, markersize=8)
    ax.set_xlabel('Corpus Size', fontsize=12)
    ax.set_ylabel('Similarity per Minute', fontsize=12)
    ax.set_title('Training Efficiency: Similarity Gain per Minute', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_efficiency.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: {output_dir}/training_efficiency.png")
    
    # Plot 3: Non-linear methods comparison
    if nonlinear_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = list(nonlinear_results.keys())
        val_sims = [nonlinear_results[m]["validation_metrics"]["mean"] for m in methods]
        test_sims = [nonlinear_results[m]["test_metrics"]["mean"] for m in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        ax.bar(x - width/2, val_sims, width, label='Validation', alpha=0.8)
        ax.bar(x + width/2, test_sims, width, label='Test', alpha=0.8)
        
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Similarity', fontsize=12)
        ax.set_title('Non-Linear Methods Comparison', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/nonlinear_methods_comparison.png", dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot: {output_dir}/nonlinear_methods_comparison.png")
    
    plt.close('all')


def main():
    """Main experiment runner."""
    print("="*70)
    print(" OPTIMIZATION EXPERIMENTS")
    print("="*70)
    print("\nExperiments:")
    print("  1. Corpus size optimization (find sweet spot)")
    print("  2. Non-linear transformation methods")
    print()
    
    # Load embedding models
    print("Loading embedding models...")
    embedder1 = SentenceTransformer('all-MiniLM-L6-v2')  # 384d
    embedder2 = SentenceTransformer('all-mpnet-base-v2')  # 768d
    print("✓ Models loaded\n")
    
    # Experiment 1: Corpus size sweep
    # Smaller range for faster experiments - will terminate early if no improvement
    corpus_sizes = [1000, 2000, 5000, 10000, 20000]
    print(f"Testing corpus sizes: {corpus_sizes}")
    print("(Will terminate early if 3+ consecutive sizes show no efficiency gain)")
    
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
    print(f"Optimal corpus size: {optimal_analysis['optimal_corpus_size']:,}")
    print(f"  Similarity: {optimal_analysis['optimal_similarity']:.4f}")
    print(f"  Training time: {optimal_analysis['optimal_time']/60:.2f} minutes")
    
    # Experiment 2: Non-linear methods (use optimal corpus size)
    optimal_size = optimal_analysis['optimal_corpus_size']
    print(f"\n{'='*70}")
    print(f"Testing non-linear methods with corpus size: {optimal_size:,}")
    print(f"{'='*70}")
    
    # Generate vocabulary
    full_vocab = generate_diverse_vocabulary(optimal_size + 5000 + 1000)
    train_vocab = full_vocab[:optimal_size]
    val_vocab = full_vocab[optimal_size:optimal_size+5000]
    test_vocab = full_vocab[optimal_size+5000:optimal_size+6000]
    
    nonlinear_results = experiment_nonlinear_methods(
        embedder1, embedder2,
        train_vocab, val_vocab, test_vocab
    )
    
    # Save results
    os.makedirs("reports", exist_ok=True)
    
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "corpus_size_experiment": corpus_results,
        "optimal_analysis": optimal_analysis,
        "nonlinear_experiment": nonlinear_results
    }
    
    with open("reports/optimization_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Saved results: reports/optimization_results.json")
    
    # Generate plots
    plot_results(corpus_results, nonlinear_results)
    
    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    print("\nCorpus Size Optimization:")
    for size in sorted(corpus_results.keys()):
        r = corpus_results[size]
        print(f"  {size:>8,}: {r['validation_similarity']:.4f} similarity, "
              f"{r['training_time_seconds']/60:>6.2f} min")
    
    print("\nNon-Linear Methods:")
    for method in nonlinear_results.keys():
        r = nonlinear_results[method]
        print(f"  {method:>12}: {r['validation_metrics']['mean']:.4f} validation, "
              f"{r['test_metrics']['mean']:.4f} test, "
              f"{r['training_time_seconds']/60:>6.2f} min")
    
    print("\n✓ Experiments complete!")


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
