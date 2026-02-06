use ndarray::Array1;
use num_complex::Complex64;
use rayon::prelude::*;

/// Compute the cosine similarity between two complex vectors.
///
/// For complex vectors, cosine similarity is defined as:
///   Re(⟨a, b⟩) / (‖a‖ · ‖b‖)
///
/// where ⟨a, b⟩ is the Hermitian inner product (conjugate of first argument).
pub fn cosine_similarity(a: &Array1<Complex64>, b: &Array1<Complex64>) -> f64 {
    let dot: Complex64 = a
        .iter()
        .zip(b.iter())
        .map(|(ai, bi)| ai.conj() * bi)
        .sum();

    let norm_a: f64 = a.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();

    if norm_a < 1e-15 || norm_b < 1e-15 {
        return 0.0;
    }

    // Clamp to [-1, 1] to handle floating-point rounding that can push
    // the result slightly outside the valid range for cosine similarity.
    (dot.re / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

/// Find the top-k most similar vectors to a query vector.
///
/// Returns a vector of (index, similarity_score) pairs, sorted in
/// descending order of similarity. Only candidates with similarity
/// at or above `threshold` are included.
pub fn find_similar(
    query: &Array1<Complex64>,
    candidates: &[Array1<Complex64>],
    threshold: f64,
    top_k: usize,
) -> Vec<(usize, f64)> {
    let mut scores: Vec<(usize, f64)> = candidates
        .par_iter()
        .enumerate()
        .map(|(i, c)| (i, cosine_similarity(query, c)))
        .filter(|(_, score)| *score >= threshold)
        .collect();

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(top_k);
    scores
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::VsaPrimitives;

    #[test]
    fn self_similarity_is_one() {
        let prims = VsaPrimitives::new(512, 42);
        let sim = cosine_similarity(&prims.negative_qty, &prims.negative_qty);
        assert!(
            (sim - 1.0).abs() < 1e-10,
            "self-similarity should be 1.0, got {}",
            sim
        );
    }

    #[test]
    fn different_primitives_near_orthogonal() {
        // In high dimensions, random phasor vectors are nearly orthogonal.
        let prims = VsaPrimitives::new(4096, 42);
        let sim = cosine_similarity(&prims.negative_qty, &prims.high_cost);
        assert!(
            sim.abs() < 0.1,
            "random phasor vectors in 4096-d should be nearly orthogonal, got {}",
            sim
        );
    }

    #[test]
    fn find_similar_returns_sorted_results() {
        let prims = VsaPrimitives::new(1024, 42);
        let candidates = vec![
            prims.negative_qty.clone(),
            prims.high_cost.clone(),
            prims.low_margin.clone(),
        ];
        let results = find_similar(&prims.negative_qty, &candidates, -1.0, 10);
        // First result should be the identical vector (index 0) with similarity ~1.0
        assert_eq!(results[0].0, 0);
        assert!((results[0].1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn find_similar_respects_threshold() {
        let prims = VsaPrimitives::new(1024, 42);
        let candidates = vec![
            prims.negative_qty.clone(),
            prims.high_cost.clone(),
        ];
        // Only the self-match should pass a threshold of 0.5
        let results = find_similar(&prims.negative_qty, &candidates, 0.5, 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn find_similar_respects_top_k() {
        let prims = VsaPrimitives::new(256, 42);
        let candidates = vec![
            prims.negative_qty.clone(),
            prims.high_cost.clone(),
            prims.low_margin.clone(),
            prims.zero_sales.clone(),
        ];
        let results = find_similar(&prims.negative_qty, &candidates, -1.0, 2);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn zero_vector_similarity_is_zero() {
        let prims = VsaPrimitives::new(256, 42);
        let zero = Array1::<Complex64>::zeros(256);
        let sim = cosine_similarity(&zero, &prims.negative_qty);
        assert!((sim).abs() < 1e-10, "zero vector similarity should be 0");
    }
}
