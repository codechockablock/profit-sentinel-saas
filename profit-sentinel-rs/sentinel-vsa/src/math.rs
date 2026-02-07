//! Shared mathematical utilities for VSA operations.

use ndarray::Array1;
use num_complex::Complex64;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f64::consts::PI;

/// FNV-1a hash for deterministic seed generation.
pub fn fnv1a_hash(data: &[u8]) -> u64 {
    let mut hash: u64 = 14695981039346656037;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    hash
}

/// Generate a random phasor vector (unit-magnitude complex components)
/// from a mutable RNG. Use this when generating multiple vectors from
/// a single seeded RNG sequence (e.g., VsaPrimitives).
pub fn random_phasor_vector(dimensions: usize, rng: &mut StdRng) -> Array1<Complex64> {
    Array1::from_iter(
        (0..dimensions).map(|_| Complex64::from_polar(1.0, rng.gen_range(0.0..2.0 * PI))),
    )
}

/// Generate a random phasor vector from a u64 seed. Use this when each
/// vector needs its own independent, reproducible seed (e.g., CauseVectors).
pub fn random_phasor_vector_from_seed(dimensions: usize, seed: u64) -> Array1<Complex64> {
    let mut rng = StdRng::seed_from_u64(seed);
    random_phasor_vector(dimensions, &mut rng)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv1a_is_deterministic() {
        let h1 = fnv1a_hash(b"cause_vector_theft_v1");
        let h2 = fnv1a_hash(b"cause_vector_theft_v1");
        assert_eq!(h1, h2);
    }

    #[test]
    fn different_inputs_produce_different_hashes() {
        let h1 = fnv1a_hash(b"cause_vector_theft_v1");
        let h2 = fnv1a_hash(b"cause_vector_demand_shift_v1");
        assert_ne!(h1, h2);
    }

    #[test]
    fn phasor_vector_has_correct_dimensions() {
        let vec = random_phasor_vector_from_seed(512, 42);
        assert_eq!(vec.len(), 512);
    }

    #[test]
    fn phasor_vector_has_unit_magnitude_components() {
        let vec = random_phasor_vector_from_seed(256, 99);
        for &c in vec.iter() {
            let mag = c.norm();
            assert!((mag - 1.0).abs() < 1e-10, "magnitude was {}", mag);
        }
    }

    #[test]
    fn phasor_vector_from_seed_is_deterministic() {
        let v1 = random_phasor_vector_from_seed(256, 42);
        let v2 = random_phasor_vector_from_seed(256, 42);
        assert_eq!(v1, v2);
    }

    #[test]
    fn different_seeds_produce_different_vectors() {
        let v1 = random_phasor_vector_from_seed(256, 1);
        let v2 = random_phasor_vector_from_seed(256, 2);
        assert_ne!(v1, v2);
    }

    #[test]
    fn rng_variant_matches_seed_variant() {
        // Verify that random_phasor_vector with a manually seeded RNG
        // produces the same result as random_phasor_vector_from_seed.
        let mut rng = StdRng::seed_from_u64(42);
        let v1 = random_phasor_vector(256, &mut rng);
        let v2 = random_phasor_vector_from_seed(256, 42);
        assert_eq!(v1, v2);
    }
}
