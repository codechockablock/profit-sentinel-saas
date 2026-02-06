use ndarray::Array1;
use num_complex::Complex64;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f64::consts::PI;

/// The 11 VSA primitive vectors used to encode inventory anomaly signals.
///
/// Each primitive is a random phasor vector (unit-magnitude complex numbers)
/// generated from a seeded RNG so that results are reproducible. The vectors
/// are approximately orthogonal in high dimensions due to the concentration
/// of measure phenomenon.
pub struct VsaPrimitives {
    pub negative_qty: Array1<Complex64>,
    pub high_cost: Array1<Complex64>,
    pub low_margin: Array1<Complex64>,
    pub zero_sales: Array1<Complex64>,
    pub high_qty: Array1<Complex64>,
    pub recent_receipt: Array1<Complex64>,
    pub old_receipt: Array1<Complex64>,
    pub negative_retail: Array1<Complex64>,
    pub damaged: Array1<Complex64>,
    pub on_order: Array1<Complex64>,
    pub seasonal: Array1<Complex64>,
    dimensions: usize,
}

impl VsaPrimitives {
    /// Create a new set of primitives with the given dimensionality and seed.
    ///
    /// Each primitive is a random phasor vector: every component has magnitude 1
    /// with a uniformly random phase angle in [0, 2π).
    pub fn new(dimensions: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        Self {
            negative_qty: random_phasor_vector(dimensions, &mut rng),
            high_cost: random_phasor_vector(dimensions, &mut rng),
            low_margin: random_phasor_vector(dimensions, &mut rng),
            zero_sales: random_phasor_vector(dimensions, &mut rng),
            high_qty: random_phasor_vector(dimensions, &mut rng),
            recent_receipt: random_phasor_vector(dimensions, &mut rng),
            old_receipt: random_phasor_vector(dimensions, &mut rng),
            negative_retail: random_phasor_vector(dimensions, &mut rng),
            damaged: random_phasor_vector(dimensions, &mut rng),
            on_order: random_phasor_vector(dimensions, &mut rng),
            seasonal: random_phasor_vector(dimensions, &mut rng),
            dimensions,
        }
    }

    /// The dimensionality of the hypervectors.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
}

/// Generate a random phasor vector of the given length.
///
/// Each component has magnitude 1 with a uniformly random phase.
fn random_phasor_vector(dimensions: usize, rng: &mut StdRng) -> Array1<Complex64> {
    Array1::from_iter((0..dimensions).map(|_| {
        let phase: f64 = rng.gen_range(0.0..2.0 * PI);
        Complex64::from_polar(1.0, phase)
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primitives_have_correct_dimensions() {
        let p = VsaPrimitives::new(1024, 42);
        assert_eq!(p.negative_qty.len(), 1024);
        assert_eq!(p.seasonal.len(), 1024);
        assert_eq!(p.dimensions(), 1024);
    }

    #[test]
    fn phasor_vectors_have_unit_magnitude() {
        let p = VsaPrimitives::new(512, 99);
        for &c in p.negative_qty.iter() {
            let mag = c.norm();
            assert!((mag - 1.0).abs() < 1e-10, "magnitude was {}", mag);
        }
    }

    #[test]
    fn deterministic_with_same_seed() {
        let a = VsaPrimitives::new(256, 42);
        let b = VsaPrimitives::new(256, 42);
        assert_eq!(a.negative_qty, b.negative_qty);
        assert_eq!(a.seasonal, b.seasonal);
    }

    #[test]
    fn different_seeds_produce_different_vectors() {
        let a = VsaPrimitives::new(256, 1);
        let b = VsaPrimitives::new(256, 2);
        assert_ne!(a.negative_qty, b.negative_qty);
    }
}
