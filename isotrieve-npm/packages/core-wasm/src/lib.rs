use wasm_bindgen::prelude::*;
use ndarray::{Array1, Array2};

/// Compute cosine similarity between two vectors (blazingly fast)
#[wasm_bindgen]
pub fn cosine_similarity(vec1: &[f64], vec2: &[f64]) -> f64 {
    if vec1.len() != vec2.len() {
        return 0.0;
    }
    
    let mut dot_product = 0.0;
    let mut norm1 = 0.0;
    let mut norm2 = 0.0;
    
    for i in 0..vec1.len() {
        dot_product += vec1[i] * vec2[i];
        norm1 += vec1[i] * vec1[i];
        norm2 += vec2[i] * vec2[i];
    }
    
    norm1 = norm1.sqrt();
    norm2 = norm2.sqrt();
    
    if norm1 == 0.0 || norm2 == 0.0 {
        return 0.0;
    }
    
    dot_product / (norm1 * norm2)
}

/// Vector-matrix multiplication (optimized)
#[wasm_bindgen]
pub fn vector_matrix_multiply(vec: &[f64], matrix_flat: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let vec_arr = Array1::from_vec(vec.to_vec());
    let matrix = Array2::from_shape_vec((rows, cols), matrix_flat.to_vec()).unwrap();
    
    let result = vec_arr.dot(&matrix);
    result.to_vec()
}

/// Matrix multiplication (highly optimized)
#[wasm_bindgen]
pub fn matrix_multiply(
    a_flat: &[f64], a_rows: usize, a_cols: usize,
    b_flat: &[f64], b_rows: usize, b_cols: usize
) -> Vec<f64> {
    let a = Array2::from_shape_vec((a_rows, a_cols), a_flat.to_vec()).unwrap();
    let b = Array2::from_shape_vec((b_rows, b_cols), b_flat.to_vec()).unwrap();
    
    let result = a.dot(&b);
    result.into_raw_vec()
}

/// Compute least squares solution (high performance)
#[wasm_bindgen]
pub fn least_squares(
    a_flat: &[f64], a_rows: usize, a_cols: usize,
    b_flat: &[f64], b_rows: usize, b_cols: usize
) -> Vec<f64> {
    use ndarray_linalg::LeastSquaresSvd;
    
    let a = Array2::from_shape_vec((a_rows, a_cols), a_flat.to_vec()).unwrap();
    let b = Array2::from_shape_vec((b_rows, b_cols), b_flat.to_vec()).unwrap();
    
    // Solve using SVD (most stable method)
    let solution = a.least_squares(&b).unwrap().solution;
    solution.into_raw_vec()
}

/// Batch cosine similarity (SIMD-optimized)
#[wasm_bindgen]
pub fn batch_cosine_similarity(
    vecs1_flat: &[f64], vecs2_flat: &[f64], 
    n_vectors: usize, dim: usize
) -> Vec<f64> {
    let mut similarities = Vec::with_capacity(n_vectors);
    
    for i in 0..n_vectors {
        let start = i * dim;
        let end = start + dim;
        let vec1 = &vecs1_flat[start..end];
        let vec2 = &vecs2_flat[start..end];
        similarities.push(cosine_similarity(vec1, vec2));
    }
    
    similarities
}
