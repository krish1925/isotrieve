"""
AECP Matrix Operations

Core matrix operations for embedding transfer between agents.
Implements least-squares linear transformation for embedding space alignment.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
import warnings


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector (any shape, will be flattened)
        vec2: Second vector (any shape, will be flattened)
        
    Returns:
        Cosine similarity score in range [-1, 1]
    """
    vec1 = np.asarray(vec1).flatten()
    vec2 = np.asarray(vec2).flatten()
    
    if vec1.size != vec2.size:
        raise ValueError(
            f"Vectors must have same size: {vec1.size} vs {vec2.size}"
        )
    
    if vec1.size == 0:
        raise ValueError("Vectors cannot be empty")
    
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    # Handle zero vectors gracefully
    if norm1 < 1e-12 or norm2 < 1e-12:
        return 0.0
    
    dot_product = np.dot(vec1, vec2)
    similarity = dot_product / (norm1 * norm2)
    
    # Clamp to [-1, 1] to handle floating point errors
    return float(np.clip(similarity, -1.0, 1.0))


def compute_transfer_matrices(
    embeddings_source: np.ndarray,
    embeddings_target: np.ndarray,
    method: str = "ridge",
    regularization: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute transfer matrices between two embedding spaces.
    
    Uses ridge regression to find linear transformations that
    map embeddings from one space to another.
    
    Args:
        embeddings_source: Source embeddings [n_samples, dim_source]
        embeddings_target: Target embeddings [n_samples, dim_target]
        method: Method for computing matrices ('lstsq' or 'ridge')
        regularization: Regularization strength for ridge regression
        
    Returns:
        Tuple of (W_source_to_target, W_target_to_source)
    """
    # Input validation
    embeddings_source = np.asarray(embeddings_source, dtype=np.float64)
    embeddings_target = np.asarray(embeddings_target, dtype=np.float64)
    
    if embeddings_source.ndim != 2 or embeddings_target.ndim != 2:
        raise ValueError("Embeddings must be 2-dimensional arrays")
    
    if embeddings_source.shape[0] != embeddings_target.shape[0]:
        raise ValueError(
            f"Number of samples must match: "
            f"{embeddings_source.shape[0]} vs {embeddings_target.shape[0]}"
        )
    
    n_samples = embeddings_source.shape[0]
    if n_samples == 0:
        raise ValueError("Cannot compute transfer matrices with zero samples")
    
    dim_source = embeddings_source.shape[1]
    dim_target = embeddings_target.shape[1]
    
    if method == "lstsq":
        try:
            # Add tiny regularization to prevent blowup even in lstsq
            W_source_to_target = np.linalg.lstsq(
                embeddings_source, 
                embeddings_target, 
                rcond=1e-10
            )[0]
            
            W_target_to_source = np.linalg.lstsq(
                embeddings_target,
                embeddings_source,
                rcond=1e-10
            )[0]
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"Matrix computation failed: {e}")
            
    elif method == "ridge":
        try:
            # W = (X^T X + λI)^(-1) X^T Y
            # Compute X^T X
            XtX_source = embeddings_source.T @ embeddings_source
            XtY_source = embeddings_source.T @ embeddings_target
            
            # Add regularization
            reg_source = regularization * np.eye(dim_source) * (n_samples / 1000.0)
            W_source_to_target = np.linalg.solve(XtX_source + reg_source, XtY_source)
            
            XtX_target = embeddings_target.T @ embeddings_target
            XtY_target = embeddings_target.T @ embeddings_source
            reg_target = regularization * np.eye(dim_target) * (n_samples / 1000.0)
            W_target_to_source = np.linalg.solve(XtX_target + reg_target, XtY_target)
            
        except np.linalg.LinAlgError:
            # Fall back to lstsq if solve fails
            return compute_transfer_matrices(embeddings_source, embeddings_target, method="lstsq")
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return W_source_to_target, W_target_to_source


def transfer_embedding(
    embedding: np.ndarray,
    transfer_matrix: np.ndarray
) -> np.ndarray:
    """Transfer an embedding to a different space."""
    embedding = np.asarray(embedding, dtype=np.float64)
    transfer_matrix = np.asarray(transfer_matrix, dtype=np.float64)
    
    if transfer_matrix.ndim != 2:
        raise ValueError("Transfer matrix must be 2-dimensional")
        
    is_single = embedding.ndim == 1
    if is_single:
        if embedding.shape[0] != transfer_matrix.shape[0]:
            raise ValueError(f"Embedding size {embedding.shape[0]} does not match matrix input dimension {transfer_matrix.shape[0]}")
        embedding = embedding.reshape(1, -1)
    else:
        if embedding.shape[1] != transfer_matrix.shape[0]:
            raise ValueError(f"Embedding dimension {embedding.shape[1]} does not match matrix input dimension {transfer_matrix.shape[0]}")
    
    result = embedding @ transfer_matrix
    
    if is_single:
        result = result.flatten()
    
    return result


def evaluate_transfer_quality(
    embeddings_source: np.ndarray,
    embeddings_target: np.ndarray,
    W_forward: np.ndarray,
    W_backward: Optional[np.ndarray] = None,
    sample_size: Optional[int] = None
) -> Dict[str, float]:
    """Evaluate transfer matrix quality."""
    embeddings_source = np.asarray(embeddings_source, dtype=np.float64)
    embeddings_target = np.asarray(embeddings_target, dtype=np.float64)
    
    n_samples = len(embeddings_source)
    if sample_size is not None and sample_size < n_samples:
        indices = np.random.choice(n_samples, sample_size, replace=False)
        embeddings_source = embeddings_source[indices]
        embeddings_target = embeddings_target[indices]
        n_samples = sample_size
    
    transferred = embeddings_source @ W_forward
    
    f_sims = []
    for i in range(n_samples):
        f_sims.append(cosine_similarity(transferred[i], embeddings_target[i]))
    
    f_sims = np.array(f_sims)
    
    results = {
        "forward_mean_similarity": float(np.mean(f_sims)),
        "forward_min_similarity": float(np.min(f_sims)),
    }
    
    if W_backward is not None:
        roundtrip = transferred @ W_backward
        r_sims = []
        for i in range(n_samples):
            r_sims.append(cosine_similarity(embeddings_source[i], roundtrip[i]))
        r_sims = np.array(r_sims)
        results.update({
            "roundtrip_mean_similarity": float(np.mean(r_sims)),
            "roundtrip_min_similarity": float(np.min(r_sims)),
        })
    
    return results


def compute_embedding_stats(embeddings: np.ndarray) -> Dict:
    """Compute statistics for a set of embeddings."""
    embeddings = np.asarray(embeddings, dtype=np.float64)
    n_samples, dimensions = embeddings.shape
    
    norms = np.linalg.norm(embeddings, axis=1)
    
    # Pairwise similarity is expensive for large sets, sample if needed
    if n_samples > 100:
        indices = np.random.choice(n_samples, 100, replace=False)
        subset = embeddings[indices]
    else:
        subset = embeddings
        
    # Normalized subset for similarity
    subset_norms = np.linalg.norm(subset, axis=1, keepdims=True)
    subset_norms[subset_norms < 1e-12] = 1.0
    subset_normed = subset / subset_norms
    
    sim_matrix = subset_normed @ subset_normed.T
    # Exclude diagonal
    mask = ~np.eye(sim_matrix.shape[0], dtype=bool)
    mean_pairwise = float(np.mean(sim_matrix[mask])) if n_samples > 1 else 1.0
    
    # Numerical rank
    try:
        rank = int(np.linalg.matrix_rank(embeddings, tol=1e-10))
    except:
        rank = 0
        
    return {
        "n_samples": n_samples,
        "dimensions": dimensions,
        "mean_norm": float(np.mean(norms)),
        "std_norm": float(np.std(norms)),
        "mean_pairwise_similarity": mean_pairwise,
        "matrix_rank": rank,
    }
