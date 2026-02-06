use ndarray::Array1;
use num_complex::Complex64;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;
use std::sync::{Arc, RwLock};

/// A codebook maps SKU identifiers to deterministic phasor vectors.
///
/// Vectors are lazily generated on first access and cached for reuse.
/// A consistent seed is derived from the SKU string so that the same
/// SKU always maps to the same vector regardless of access order.
///
/// Uses `Arc<Array1<Complex64>>` for zero-copy sharing across threads
/// and `RwLock` for concurrent reads during the parallel bundling phase.
pub struct Codebook {
    dimensions: usize,
    base_seed: u64,
    cache: RwLock<HashMap<String, Arc<Array1<Complex64>>>>,
}

impl Codebook {
    pub fn new(dimensions: usize, base_seed: u64) -> Self {
        Self {
            dimensions,
            base_seed,
            cache: RwLock::new(HashMap::new()),
        }
    }

    /// Get or create the vector for a given SKU.
    ///
    /// Returns an `Arc` to avoid cloning the 1024-element array.
    /// Fast path: read lock for cache hit (allows concurrent reads).
    /// Slow path: write lock only on cache miss.
    pub fn get_or_create(&self, sku: &str) -> Arc<Array1<Complex64>> {
        // Fast path: read-only lock for cache hits
        {
            let cache = self.cache.read().unwrap();
            if let Some(v) = cache.get(sku) {
                return Arc::clone(v);
            }
        }
        // Slow path: generate and insert under write lock
        let vec = Arc::new(self.generate(sku));
        let mut cache = self.cache.write().unwrap();
        // Double-check in case another thread inserted while we waited
        Arc::clone(cache.entry(sku.to_string()).or_insert(vec))
    }

    /// Pre-populate the codebook for a set of SKUs.
    ///
    /// Call this before the parallel bundling phase to avoid lock
    /// contention during the hot path. After warmup, all `get_or_create`
    /// calls will hit the fast read-lock path.
    pub fn warmup<'a, I: IntoIterator<Item = &'a str>>(&self, skus: I) {
        let mut cache = self.cache.write().unwrap();
        for sku in skus {
            if !cache.contains_key(sku) {
                let vec = Arc::new(self.generate(sku));
                cache.insert(sku.to_string(), vec);
            }
        }
    }

    /// Pre-populate the codebook in parallel using Rayon.
    ///
    /// Collects unique SKUs, generates their vectors in parallel, then
    /// inserts them all under a single write lock. This turns the O(N)
    /// sequential warmup into an O(N/cores) parallel phase.
    pub fn warmup_parallel<'a, I: IntoIterator<Item = &'a str>>(&self, skus: I) {
        // Deduplicate SKUs and filter out already-cached ones
        let cache = self.cache.read().unwrap();
        let unique: Vec<String> = skus
            .into_iter()
            .collect::<HashSet<&str>>()
            .into_iter()
            .filter(|s| !cache.contains_key(*s))
            .map(|s| s.to_string())
            .collect();
        drop(cache);

        if unique.is_empty() {
            return;
        }

        // Generate vectors in parallel
        let generated: Vec<(String, Arc<Array1<Complex64>>)> = unique
            .par_iter()
            .map(|sku| {
                let vec = Arc::new(self.generate(sku));
                (sku.clone(), vec)
            })
            .collect();

        // Insert all under a single write lock
        let mut cache = self.cache.write().unwrap();
        for (sku, vec) in generated {
            cache.entry(sku).or_insert(vec);
        }
    }

    /// Generate a deterministic phasor vector for a SKU.
    fn generate(&self, sku: &str) -> Array1<Complex64> {
        let seed = self.sku_seed(sku);
        let mut rng = StdRng::seed_from_u64(seed);
        Array1::from_iter((0..self.dimensions).map(|_| {
            let phase: f64 = rng.gen_range(0.0..2.0 * PI);
            Complex64::from_polar(1.0, phase)
        }))
    }

    /// Derive a deterministic seed from the SKU string and the base seed.
    fn sku_seed(&self, sku: &str) -> u64 {
        // Simple FNV-1a-style hash combined with the base seed.
        let mut hash: u64 = 14695981039346656037;
        for byte in sku.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(1099511628211);
        }
        hash ^ self.base_seed
    }

    /// Take a lock-free snapshot of the codebook.
    ///
    /// Returns a plain `HashMap` that can be shared across threads with
    /// zero synchronization overhead. Call `warmup()` first to ensure
    /// all needed SKUs are in the cache.
    pub fn snapshot(&self) -> HashMap<String, Arc<Array1<Complex64>>> {
        self.cache.read().unwrap().clone()
    }

    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    pub fn len(&self) -> usize {
        self.cache.read().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_sku_returns_same_vector() {
        let cb = Codebook::new(256, 42);
        let v1 = cb.get_or_create("SKU-001");
        let v2 = cb.get_or_create("SKU-001");
        assert_eq!(*v1, *v2);
    }

    #[test]
    fn different_skus_return_different_vectors() {
        let cb = Codebook::new(256, 42);
        let v1 = cb.get_or_create("SKU-001");
        let v2 = cb.get_or_create("SKU-002");
        assert_ne!(*v1, *v2);
    }

    #[test]
    fn codebook_caches_entries() {
        let cb = Codebook::new(256, 42);
        assert!(cb.is_empty());
        cb.get_or_create("SKU-001");
        assert_eq!(cb.len(), 1);
        cb.get_or_create("SKU-001");
        assert_eq!(cb.len(), 1);
        cb.get_or_create("SKU-002");
        assert_eq!(cb.len(), 2);
    }

    #[test]
    fn warmup_populates_cache() {
        let cb = Codebook::new(256, 42);
        cb.warmup(["SKU-001", "SKU-002", "SKU-003"].iter().copied());
        assert_eq!(cb.len(), 3);
        // Verify warmup produces same vectors as get_or_create
        let v = cb.get_or_create("SKU-001");
        let cb2 = Codebook::new(256, 42);
        let v2 = cb2.get_or_create("SKU-001");
        assert_eq!(*v, *v2);
    }
}
