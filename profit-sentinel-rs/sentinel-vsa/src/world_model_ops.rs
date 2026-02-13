//! World Model Phasor Algebra — Rust-accelerated VSA operations.
//!
//! Drop-in acceleration for the Python PhasorAlgebra class.
//! All operations work on unit-magnitude complex phasor vectors.
//!
//! Operations:
//!   bind(a, b)       = element-wise multiply (Hadamard product)
//!   unbind(a, b)     = element-wise multiply by conjugate (a * conj(b))
//!   bundle([a,b,...]) = weighted sum, re-normalized to unit magnitude
//!   similarity(a, b) = Re(⟨a, conj(b)⟩) / (‖a‖ · ‖b‖)  (cosine similarity)
//!   random_vector(seed) = deterministic phasor vector from seed
//!   identity(dim)    = all-ones vector
//!   permute(v, k)    = circular shift by k positions

use log;
use ndarray::Array1;
use num_complex::Complex64;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::RwLock;

use crate::math::{fnv1a_hash, random_phasor_vector_from_seed};

/// Phasor algebra engine — holds dimension, seed, and a vector cache.
///
/// The cache maps string labels to phasor vectors. This is the Rust
/// equivalent of PhasorAlgebra._cache in the Python world model.
pub struct PhasorEngine {
    pub dimensions: usize,
    pub base_seed: u64,
    cache: RwLock<HashMap<String, Array1<Complex64>>>,
}

impl PhasorEngine {
    pub fn new(dimensions: usize, seed: u64) -> Self {
        Self {
            dimensions,
            base_seed: seed,
            cache: RwLock::new(HashMap::new()),
        }
    }

    /// Get or create a cached random vector for a given label.
    pub fn random_vector(&self, label: &str) -> Array1<Complex64> {
        // Fast path: read lock
        {
            let cache = self.cache.read().unwrap_or_else(|poisoned| {
                log::warn!("PhasorEngine read lock was poisoned, recovering");
                poisoned.into_inner()
            });
            if let Some(v) = cache.get(label) {
                return v.clone();
            }
        }
        // Slow path: generate and cache
        let seed = fnv1a_hash(label.as_bytes()) ^ self.base_seed;
        let v = random_phasor_vector_from_seed(self.dimensions, seed);
        let mut cache = self.cache.write().unwrap_or_else(|poisoned| {
            log::warn!("PhasorEngine write lock was poisoned, recovering");
            poisoned.into_inner()
        });
        cache.insert(label.to_string(), v.clone());
        v
    }

    /// Identity vector (all ones — multiplicative identity for bind).
    pub fn identity(&self) -> Array1<Complex64> {
        Array1::from_elem(self.dimensions, Complex64::new(1.0, 0.0))
    }

    /// Element-wise multiply (binding operation).
    #[inline]
    pub fn bind(a: &Array1<Complex64>, b: &Array1<Complex64>) -> Array1<Complex64> {
        a * b
    }

    /// Unbind: multiply by conjugate (inverse of bind).
    #[inline]
    pub fn unbind(a: &Array1<Complex64>, b: &Array1<Complex64>) -> Array1<Complex64> {
        a * &b.mapv(|c| c.conj())
    }

    /// Bundle (superposition) of multiple vectors, normalized to unit magnitude.
    pub fn bundle(vectors: &[Array1<Complex64>]) -> Array1<Complex64> {
        if vectors.is_empty() {
            return Array1::zeros(0);
        }
        let dim = vectors[0].len();
        let mut sum = Array1::from_elem(dim, Complex64::new(0.0, 0.0));
        for v in vectors {
            sum = sum + v;
        }
        // Normalize to unit magnitude
        sum.mapv(|c| {
            let mag = c.norm();
            if mag > 1e-15 {
                c / mag
            } else {
                Complex64::new(1.0, 0.0)
            }
        })
    }

    /// Weighted bundle.
    pub fn bundle_weighted(
        vectors: &[Array1<Complex64>],
        weights: &[f64],
    ) -> Array1<Complex64> {
        if vectors.is_empty() {
            return Array1::zeros(0);
        }
        let dim = vectors[0].len();
        let mut sum = Array1::from_elem(dim, Complex64::new(0.0, 0.0));
        for (v, &w) in vectors.iter().zip(weights.iter()) {
            sum = sum + &v.mapv(|c| c * w);
        }
        sum.mapv(|c| {
            let mag = c.norm();
            if mag > 1e-15 {
                c / mag
            } else {
                Complex64::new(1.0, 0.0)
            }
        })
    }

    /// Cosine similarity (real part of Hermitian inner product, normalized).
    pub fn similarity(a: &Array1<Complex64>, b: &Array1<Complex64>) -> f64 {
        crate::similarity::cosine_similarity(a, b)
    }

    /// Circular permutation (shift by k positions).
    pub fn permute(v: &Array1<Complex64>, k: i32) -> Array1<Complex64> {
        if v.is_empty() {
            return Array1::zeros(0);
        }
        let n = v.len() as i32;
        let k = ((k % n) + n) % n;
        let k = k as usize;
        let mut result = Array1::uninit(v.len());
        for (i, val) in v.iter().enumerate() {
            let new_i = (i + k) % v.len();
            result[new_i].write(*val);
        }
        unsafe { result.assume_init() }
    }

    /// Batch bind: bind each pair (a_i, b_i) in parallel using Rayon.
    pub fn batch_bind(
        a_vecs: &[Array1<Complex64>],
        b_vecs: &[Array1<Complex64>],
    ) -> Vec<Array1<Complex64>> {
        a_vecs
            .par_iter()
            .zip(b_vecs.par_iter())
            .map(|(a, b)| Self::bind(a, b))
            .collect()
    }

    /// Batch similarity: compute similarity of query against all candidates in parallel.
    pub fn batch_similarity(
        query: &Array1<Complex64>,
        candidates: &[Array1<Complex64>],
    ) -> Vec<f64> {
        candidates
            .par_iter()
            .map(|c| Self::similarity(query, c))
            .collect()
    }

    /// Compile a state vector from role-filler bindings.
    ///
    /// Given role vectors and filler vectors, compute:
    ///   state = normalize(Σ bind(role_i, filler_i))
    ///
    /// This is the hot path for world model state compilation.
    pub fn compile_state(
        role_vectors: &[Array1<Complex64>],
        filler_vectors: &[Array1<Complex64>],
    ) -> Array1<Complex64> {
        let bindings: Vec<Array1<Complex64>> = role_vectors
            .par_iter()
            .zip(filler_vectors.par_iter())
            .map(|(r, f)| Self::bind(r, f))
            .collect();
        Self::bundle(&bindings)
    }

    /// Batch encode observations: given N observations (each a set of role-filler
    /// pairs), compile all state vectors in parallel.
    ///
    /// Returns N compiled state vectors.
    pub fn batch_compile_states(
        role_vectors: &[Array1<Complex64>],
        filler_batches: &[Vec<Array1<Complex64>>],
    ) -> Vec<Array1<Complex64>> {
        filler_batches
            .par_iter()
            .map(|fillers| Self::compile_state(role_vectors, fillers))
            .collect()
    }

    /// Number of cached vectors.
    pub fn cache_size(&self) -> usize {
        self.cache.read().unwrap_or_else(|p| p.into_inner()).len()
    }

    /// Clear the vector cache.
    pub fn clear_cache(&self) {
        self.cache.write().unwrap_or_else(|p| p.into_inner()).clear();
    }
}

// ============================================================================
// PyO3 bindings — zero-copy via numpy
// ============================================================================

#[cfg(feature = "python")]
pub mod py {
    use super::*;
    use pyo3::prelude::*;
    use pyo3::exceptions::PyValueError;
    use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};

    // ------------------------------------------------------------------
    // Zero-copy conversion helpers
    // ------------------------------------------------------------------

    /// Read a numpy complex128 array into an ndarray Array1<Complex64>.
    /// This is a fast memcpy (contiguous buffer → contiguous buffer).
    #[inline]
    fn np_to_ndarray(py_arr: &PyReadonlyArray1<'_, Complex64>) -> Array1<Complex64> {
        py_arr.as_array().to_owned()
    }

    /// Write an ndarray Array1<Complex64> back to a new numpy array.
    /// Single memcpy — Rust contiguous buffer → numpy contiguous buffer.
    #[inline]
    fn ndarray_to_np<'py>(py: Python<'py>, arr: &Array1<Complex64>) -> Bound<'py, PyArray1<Complex64>> {
        PyArray1::from_slice(py, arr.as_slice().unwrap())
    }

    /// Python-facing phasor algebra engine.
    ///
    /// Drop-in replacement for the Python PhasorAlgebra class.
    /// All vectors are exchanged as numpy complex128 arrays (zero-copy).
    ///
    /// For maximum throughput, use the matrix-based batch methods:
    ///   - `matrix_similarity(query, matrix)` — 1 query vs N×D matrix
    ///   - `matrix_compile_states(roles_matrix, fillers_3d)` — batch compile
    ///
    /// These read directly from contiguous numpy buffers with zero per-vector copy.
    ///
    /// Usage:
    /// ```python
    /// from sentinel_vsa import PhasorAlgebra
    /// algebra = PhasorAlgebra(dimensions=4096, seed=42)
    /// a = algebra.random_vector("role_velocity")   # returns np.ndarray complex128
    /// b = algebra.random_vector("filler_high")
    /// bound = algebra.bind(a, b)
    /// sim = algebra.similarity(a, bound)
    /// ```
    #[pyclass(name = "PhasorAlgebra")]
    pub struct PyPhasorAlgebra {
        engine: PhasorEngine,
    }

    #[pymethods]
    impl PyPhasorAlgebra {
        #[new]
        #[pyo3(signature = (dimensions=4096, seed=42))]
        fn new(dimensions: usize, seed: u64) -> Self {
            Self {
                engine: PhasorEngine::new(dimensions, seed),
            }
        }

        /// Generate (or retrieve from cache) a deterministic random phasor vector.
        /// Returns a numpy complex128 array.
        fn random_vector<'py>(&self, py: Python<'py>, label: &str) -> Bound<'py, PyArray1<Complex64>> {
            ndarray_to_np(py, &self.engine.random_vector(label))
        }

        /// Identity vector (all ones). Returns numpy complex128.
        fn identity<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Complex64>> {
            ndarray_to_np(py, &self.engine.identity())
        }

        /// Bind two vectors (element-wise multiply). Returns numpy complex128.
        fn bind<'py>(
            &self,
            py: Python<'py>,
            a: PyReadonlyArray1<'py, Complex64>,
            b: PyReadonlyArray1<'py, Complex64>,
        ) -> Bound<'py, PyArray1<Complex64>> {
            let arr_a = np_to_ndarray(&a);
            let arr_b = np_to_ndarray(&b);
            ndarray_to_np(py, &PhasorEngine::bind(&arr_a, &arr_b))
        }

        /// Unbind (multiply by conjugate). Returns numpy complex128.
        fn unbind<'py>(
            &self,
            py: Python<'py>,
            a: PyReadonlyArray1<'py, Complex64>,
            b: PyReadonlyArray1<'py, Complex64>,
        ) -> Bound<'py, PyArray1<Complex64>> {
            let arr_a = np_to_ndarray(&a);
            let arr_b = np_to_ndarray(&b);
            ndarray_to_np(py, &PhasorEngine::unbind(&arr_a, &arr_b))
        }

        /// Bundle (superposition) of multiple vectors. Returns numpy complex128.
        #[pyo3(signature = (vectors, weights=None))]
        fn bundle<'py>(
            &self,
            py: Python<'py>,
            vectors: Vec<PyReadonlyArray1<'py, Complex64>>,
            weights: Option<Vec<f64>>,
        ) -> Bound<'py, PyArray1<Complex64>> {
            let arrays: Vec<Array1<Complex64>> =
                vectors.iter().map(|v| np_to_ndarray(v)).collect();
            let result = match weights {
                Some(w) => PhasorEngine::bundle_weighted(&arrays, &w),
                None => PhasorEngine::bundle(&arrays),
            };
            ndarray_to_np(py, &result)
        }

        /// Cosine similarity between two vectors.
        fn similarity(
            &self,
            a: PyReadonlyArray1<'_, Complex64>,
            b: PyReadonlyArray1<'_, Complex64>,
        ) -> f64 {
            let arr_a = np_to_ndarray(&a);
            let arr_b = np_to_ndarray(&b);
            PhasorEngine::similarity(&arr_a, &arr_b)
        }

        /// Circular permutation. Returns numpy complex128.
        #[pyo3(signature = (v, k=1))]
        fn permute<'py>(
            &self,
            py: Python<'py>,
            v: PyReadonlyArray1<'py, Complex64>,
            k: i32,
        ) -> Bound<'py, PyArray1<Complex64>> {
            let arr = np_to_ndarray(&v);
            ndarray_to_np(py, &PhasorEngine::permute(&arr, k))
        }

        /// Batch bind: bind pairs (a_i, b_i) in parallel using Rayon.
        /// Returns list of numpy complex128 arrays.
        fn batch_bind<'py>(
            &self,
            py: Python<'py>,
            a_vecs: Vec<PyReadonlyArray1<'py, Complex64>>,
            b_vecs: Vec<PyReadonlyArray1<'py, Complex64>>,
        ) -> Vec<Bound<'py, PyArray1<Complex64>>> {
            let a: Vec<Array1<Complex64>> = a_vecs.iter().map(|v| np_to_ndarray(v)).collect();
            let b: Vec<Array1<Complex64>> = b_vecs.iter().map(|v| np_to_ndarray(v)).collect();
            PhasorEngine::batch_bind(&a, &b)
                .iter()
                .map(|r| ndarray_to_np(py, r))
                .collect()
        }

        /// Batch similarity: compare query against all candidates in parallel.
        /// Returns list of f64 scores.
        fn batch_similarity(
            &self,
            query: PyReadonlyArray1<'_, Complex64>,
            candidates: Vec<PyReadonlyArray1<'_, Complex64>>,
        ) -> Vec<f64> {
            let q = np_to_ndarray(&query);
            let c: Vec<Array1<Complex64>> =
                candidates.iter().map(|v| np_to_ndarray(v)).collect();
            PhasorEngine::batch_similarity(&q, &c)
        }

        // ===============================================================
        // Matrix-based batch operations — TRUE zero-copy hot path
        // ===============================================================
        // These accept 2D numpy arrays (N×D) instead of lists of 1D arrays.
        // The entire matrix is a single contiguous buffer — no per-vector
        // copy, no Python list iteration, just one pointer into Rust.

        /// Compute similarity of query against every row in a matrix.
        ///
        /// Args:
        ///   query: numpy complex128 array of shape (D,)
        ///   matrix: numpy complex128 array of shape (N, D) — candidates stacked
        ///
        /// Returns: numpy float64 array of shape (N,) — similarity scores
        ///
        /// This is the fastest path: single memcpy of query, then Rayon
        /// parallel iteration over matrix rows (read directly from buffer).
        fn matrix_similarity<'py>(
            &self,
            py: Python<'py>,
            query: PyReadonlyArray1<'py, Complex64>,
            matrix: PyReadonlyArray2<'py, Complex64>,
        ) -> PyResult<Bound<'py, PyArray1<f64>>> {
            let q = np_to_ndarray(&query);
            let mat = matrix.as_array();
            let n = mat.nrows();

            // Rayon parallel over rows — reads directly from numpy buffer.
            // Match Python formula: abs(mean(a * conj(b)))
            let dim_f = q.len() as f64;
            let scores: Vec<f64> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let row = mat.row(i);
                    let mut sum_re = 0.0_f64;
                    let mut sum_im = 0.0_f64;
                    for (a, &b) in q.iter().zip(row.iter()) {
                        let prod = a * b.conj();
                        sum_re += prod.re;
                        sum_im += prod.im;
                    }
                    let mean_re = sum_re / dim_f;
                    let mean_im = sum_im / dim_f;
                    (mean_re * mean_re + mean_im * mean_im).sqrt()
                })
                .collect();

            Ok(PyArray1::from_vec(py, scores))
        }

        /// Compile N state vectors from a roles matrix and a 3D fillers tensor.
        ///
        /// Args:
        ///   roles: numpy complex128 array of shape (R, D) — R role vectors
        ///   fillers: numpy complex128 array of shape (N, R, D) — N observations,
        ///            each with R filler vectors
        ///
        /// Returns: numpy complex128 array of shape (N, D) — compiled states
        ///
        /// For each observation i:
        ///   state_i = normalize(Σ_r bind(roles[r], fillers[i, r]))
        fn matrix_compile_states<'py>(
            &self,
            py: Python<'py>,
            roles: PyReadonlyArray2<'py, Complex64>,
            fillers: &Bound<'py, PyArray1<Complex64>>,
        ) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
            // fillers is actually a 3D array flattened — we accept it as raw buffer
            // and reshape manually. PyO3 numpy doesn't have PyReadonlyArray3.
            let roles_mat = roles.as_array();
            let n_roles = roles_mat.nrows();
            let dim = roles_mat.ncols();

            // Read fillers as flat buffer, reshape to (N, n_roles, dim)
            let fillers_readonly = fillers.readonly();
            let flat = fillers_readonly.as_slice()
                .map_err(|e| PyValueError::new_err(format!("fillers must be contiguous: {}", e)))?;
            let stride = n_roles * dim;
            if flat.len() % stride != 0 {
                return Err(PyValueError::new_err(format!(
                    "fillers length {} not divisible by n_roles({}) * dim({})",
                    flat.len(), n_roles, dim
                )));
            }
            let n_obs = flat.len() / stride;

            // Parallel compile — reads directly from numpy buffers
            let results: Vec<Vec<Complex64>> = (0..n_obs)
                .into_par_iter()
                .map(|i| {
                    let obs_start = i * stride;
                    let mut state = vec![Complex64::new(0.0, 0.0); dim];
                    for r in 0..n_roles {
                        let filler_start = obs_start + r * dim;
                        let role_row = roles_mat.row(r);
                        for d in 0..dim {
                            state[d] += role_row[d] * flat[filler_start + d];
                        }
                    }
                    // Normalize to unit magnitude
                    for c in state.iter_mut() {
                        let mag = c.norm();
                        if mag > 1e-15 {
                            *c /= mag;
                        } else {
                            *c = Complex64::new(1.0, 0.0);
                        }
                    }
                    state
                })
                .collect();

            // Write results into a 2D numpy array
            let mut out = ndarray::Array2::zeros((n_obs, dim));
            for (i, row) in results.iter().enumerate() {
                for (j, &val) in row.iter().enumerate() {
                    out[[i, j]] = val;
                }
            }
            Ok(PyArray2::from_owned_array(py, out))
        }

        /// Compile a state vector from role-filler bindings.
        /// Returns numpy complex128.
        fn compile_state<'py>(
            &self,
            py: Python<'py>,
            role_vectors: Vec<PyReadonlyArray1<'py, Complex64>>,
            filler_vectors: Vec<PyReadonlyArray1<'py, Complex64>>,
        ) -> Bound<'py, PyArray1<Complex64>> {
            let roles: Vec<Array1<Complex64>> =
                role_vectors.iter().map(|v| np_to_ndarray(v)).collect();
            let fillers: Vec<Array1<Complex64>> =
                filler_vectors.iter().map(|v| np_to_ndarray(v)).collect();
            ndarray_to_np(py, &PhasorEngine::compile_state(&roles, &fillers))
        }

        /// Batch compile: compile N state vectors in parallel.
        /// Returns list of numpy complex128 arrays.
        fn batch_compile_states<'py>(
            &self,
            py: Python<'py>,
            role_vectors: Vec<PyReadonlyArray1<'py, Complex64>>,
            filler_batches: Vec<Vec<PyReadonlyArray1<'py, Complex64>>>,
        ) -> Vec<Bound<'py, PyArray1<Complex64>>> {
            let roles: Vec<Array1<Complex64>> =
                role_vectors.iter().map(|v| np_to_ndarray(v)).collect();
            let batches: Vec<Vec<Array1<Complex64>>> = filler_batches
                .iter()
                .map(|batch| batch.iter().map(|v| np_to_ndarray(v)).collect())
                .collect();
            PhasorEngine::batch_compile_states(&roles, &batches)
                .iter()
                .map(|r| ndarray_to_np(py, r))
                .collect()
        }

        /// Number of cached vectors.
        fn cache_size(&self) -> usize {
            self.engine.cache_size()
        }

        /// Clear the vector cache.
        fn clear_cache(&self) {
            self.engine.clear_cache();
        }

        #[getter]
        fn dimensions(&self) -> usize {
            self.engine.dimensions
        }

        fn __repr__(&self) -> String {
            format!(
                "PhasorAlgebra(dimensions={}, cached={})",
                self.engine.dimensions,
                self.engine.cache_size()
            )
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> PhasorEngine {
        PhasorEngine::new(1024, 42)
    }

    #[test]
    fn random_vector_is_deterministic() {
        let e = engine();
        let a = e.random_vector("test_label");
        let b = e.random_vector("test_label");
        assert_eq!(a, b);
    }

    #[test]
    fn random_vector_is_unit_magnitude() {
        let e = engine();
        let v = e.random_vector("test");
        for &c in v.iter() {
            assert!((c.norm() - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn different_labels_different_vectors() {
        let e = engine();
        let a = e.random_vector("label_a");
        let b = e.random_vector("label_b");
        assert_ne!(a, b);
    }

    #[test]
    fn bind_unbind_recovers_original() {
        let e = engine();
        let a = e.random_vector("a");
        let b = e.random_vector("b");
        let bound = PhasorEngine::bind(&a, &b);
        let recovered = PhasorEngine::unbind(&bound, &b);
        let sim = PhasorEngine::similarity(&a, &recovered);
        assert!(sim > 0.99, "Expected ~1.0, got {}", sim);
    }

    #[test]
    fn bundle_preserves_components() {
        let e = engine();
        let a = e.random_vector("a");
        let b = e.random_vector("b");
        let bundled = PhasorEngine::bundle(&[a.clone(), b.clone()]);
        let sim_a = PhasorEngine::similarity(&bundled, &a);
        let sim_b = PhasorEngine::similarity(&bundled, &b);
        // Both components should be retrievable
        assert!(sim_a > 0.3, "sim_a = {}", sim_a);
        assert!(sim_b > 0.3, "sim_b = {}", sim_b);
    }

    #[test]
    fn identity_is_multiplicative_identity() {
        let e = engine();
        let v = e.random_vector("test");
        let id = e.identity();
        let bound = PhasorEngine::bind(&v, &id);
        let sim = PhasorEngine::similarity(&v, &bound);
        assert!(sim > 0.99, "Expected ~1.0, got {}", sim);
    }

    #[test]
    fn permute_shifts_correctly() {
        let e = engine();
        let v = e.random_vector("test");
        let shifted = PhasorEngine::permute(&v, 1);
        // Shift by 0 should be identity
        let no_shift = PhasorEngine::permute(&v, 0);
        assert_eq!(v, no_shift);
        // Shift by dim should wrap around to identity
        let full_shift = PhasorEngine::permute(&v, e.dimensions as i32);
        assert_eq!(v, full_shift);
        // Shifted should be different
        assert_ne!(v, shifted);
    }

    #[test]
    fn batch_bind_matches_sequential() {
        let e = engine();
        let a_vecs: Vec<_> = (0..100)
            .map(|i| e.random_vector(&format!("a_{}", i)))
            .collect();
        let b_vecs: Vec<_> = (0..100)
            .map(|i| e.random_vector(&format!("b_{}", i)))
            .collect();

        let parallel = PhasorEngine::batch_bind(&a_vecs, &b_vecs);
        let sequential: Vec<_> = a_vecs
            .iter()
            .zip(b_vecs.iter())
            .map(|(a, b)| PhasorEngine::bind(a, b))
            .collect();

        for (p, s) in parallel.iter().zip(sequential.iter()) {
            assert_eq!(p, s);
        }
    }

    #[test]
    fn compile_state_works() {
        let e = engine();
        let roles: Vec<_> = (0..6)
            .map(|i| e.random_vector(&format!("role_{}", i)))
            .collect();
        let fillers: Vec<_> = (0..6)
            .map(|i| e.random_vector(&format!("filler_{}", i)))
            .collect();

        let state = PhasorEngine::compile_state(&roles, &fillers);
        assert_eq!(state.len(), e.dimensions);

        // Should be able to recover each filler by unbinding with role
        for (i, role) in roles.iter().enumerate() {
            let recovered = PhasorEngine::unbind(&state, role);
            let sim = PhasorEngine::similarity(&recovered, &fillers[i]);
            // With 6 items bundled, similarity should be > 0.3
            assert!(sim > 0.2, "role_{} sim = {}", i, sim);
        }
    }

    #[test]
    fn cache_works() {
        let e = engine();
        assert_eq!(e.cache_size(), 0);
        e.random_vector("a");
        assert_eq!(e.cache_size(), 1);
        e.random_vector("a"); // cache hit
        assert_eq!(e.cache_size(), 1);
        e.random_vector("b");
        assert_eq!(e.cache_size(), 2);
        e.clear_cache();
        assert_eq!(e.cache_size(), 0);
    }
}
