//! PyO3 bindings for sentinel-vsa.
//!
//! Exposes the core VSA operations to Python:
//! - `PyInventoryRow` — Python-friendly inventory row
//! - `bundle_inventory` — bundles rows into hypervectors
//! - `find_similar_vectors` — cosine similarity search
//! - `VsaEngine` — holds primitives + codebook, reusable across calls
//!
//! Build with: `maturin develop --features python`

use pyo3::prelude::*;

use crate::bundling::{bundle_inventory_batch, InventoryRow};
use crate::codebook::Codebook;
use crate::primitives::VsaPrimitives;
use crate::similarity;

use ndarray::Array1;
use num_complex::Complex64;

// ---------------------------------------------------------------------------
// Python-visible types
// ---------------------------------------------------------------------------

/// A single inventory row, constructable from Python.
///
/// Usage from Python:
/// ```python
/// row = PyInventoryRow(
///     sku="SKU-001",
///     qty_on_hand=-5.0,
///     unit_cost=750.0,
///     margin_pct=0.10,
///     sales_last_30d=0.0,
///     days_since_receipt=120.0,
///     retail_price=49.99,
///     is_damaged=False,
///     on_order_qty=0.0,
///     is_seasonal=False,
/// )
/// ```
#[pyclass(name = "InventoryRow", from_py_object)]
#[derive(Clone)]
pub struct PyInventoryRow {
    #[pyo3(get, set)]
    pub sku: String,
    #[pyo3(get, set)]
    pub qty_on_hand: f64,
    #[pyo3(get, set)]
    pub unit_cost: f64,
    #[pyo3(get, set)]
    pub margin_pct: f64,
    #[pyo3(get, set)]
    pub sales_last_30d: f64,
    #[pyo3(get, set)]
    pub days_since_receipt: f64,
    #[pyo3(get, set)]
    pub retail_price: f64,
    #[pyo3(get, set)]
    pub is_damaged: bool,
    #[pyo3(get, set)]
    pub on_order_qty: f64,
    #[pyo3(get, set)]
    pub is_seasonal: bool,
}

#[pymethods]
impl PyInventoryRow {
    #[new]
    #[pyo3(signature = (sku, qty_on_hand=0.0, unit_cost=0.0, margin_pct=0.0, sales_last_30d=0.0, days_since_receipt=0.0, retail_price=0.0, is_damaged=false, on_order_qty=0.0, is_seasonal=false))]
    fn new(
        sku: String,
        qty_on_hand: f64,
        unit_cost: f64,
        margin_pct: f64,
        sales_last_30d: f64,
        days_since_receipt: f64,
        retail_price: f64,
        is_damaged: bool,
        on_order_qty: f64,
        is_seasonal: bool,
    ) -> Self {
        Self {
            sku,
            qty_on_hand,
            unit_cost,
            margin_pct,
            sales_last_30d,
            days_since_receipt,
            retail_price,
            is_damaged,
            on_order_qty,
            is_seasonal,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "InventoryRow(sku='{}', qty={}, cost={:.2})",
            self.sku, self.qty_on_hand, self.unit_cost
        )
    }
}

impl From<&PyInventoryRow> for InventoryRow {
    fn from(py_row: &PyInventoryRow) -> Self {
        InventoryRow {
            sku: py_row.sku.clone(),
            qty_on_hand: py_row.qty_on_hand,
            unit_cost: py_row.unit_cost,
            margin_pct: py_row.margin_pct,
            sales_last_30d: py_row.sales_last_30d,
            days_since_receipt: py_row.days_since_receipt,
            retail_price: py_row.retail_price,
            is_damaged: py_row.is_damaged,
            on_order_qty: py_row.on_order_qty,
            is_seasonal: py_row.is_seasonal,
        }
    }
}

/// A single similarity search result.
#[pyclass(name = "SimilarityResult", from_py_object)]
#[derive(Clone)]
pub struct PySimilarityResult {
    #[pyo3(get)]
    pub index: usize,
    #[pyo3(get)]
    pub score: f64,
}

#[pymethods]
impl PySimilarityResult {
    fn __repr__(&self) -> String {
        format!(
            "SimilarityResult(index={}, score={:.4})",
            self.index, self.score
        )
    }
}

/// The VSA engine holds primitives and codebook. Create once, reuse for
/// multiple bundle/similarity calls.
///
/// Usage from Python:
/// ```python
/// engine = VsaEngine(dimensions=1024, seed=42)
/// bundles = engine.bundle_inventory(rows)
/// results = engine.find_similar(query_index=0, bundles=bundles, threshold=0.3, top_k=10)
/// ```
#[pyclass(name = "VsaEngine")]
pub struct PyVsaEngine {
    primitives: VsaPrimitives,
    codebook: Codebook,
    dimensions: usize,
}

#[pymethods]
impl PyVsaEngine {
    #[new]
    #[pyo3(signature = (dimensions=1024, seed=42))]
    fn new(dimensions: usize, seed: u64) -> Self {
        Self {
            primitives: VsaPrimitives::new(dimensions, seed),
            codebook: Codebook::new(dimensions, seed),
            dimensions,
        }
    }

    /// Bundle a list of InventoryRow objects into hypervectors.
    ///
    /// Returns a list of lists of [real, imag] pairs — one per row.
    /// Each inner list has `dimensions` elements.
    fn bundle_inventory(&self, rows: Vec<PyInventoryRow>) -> PyResult<Vec<Vec<[f64; 2]>>> {
        let rust_rows: Vec<InventoryRow> = rows.iter().map(InventoryRow::from).collect();
        let bundles = bundle_inventory_batch(&rust_rows, &self.primitives, &self.codebook);
        Ok(bundles_to_py(&bundles))
    }

    /// Find the top-k most similar vectors to a query vector.
    ///
    /// `query_index` is the index into the `bundles` list to use as the query.
    /// Returns a list of SimilarityResult objects.
    fn find_similar(
        &self,
        query_index: usize,
        bundles: Vec<Vec<[f64; 2]>>,
        threshold: f64,
        top_k: usize,
    ) -> PyResult<Vec<PySimilarityResult>> {
        let arrays = py_to_bundles(&bundles, self.dimensions)?;
        if query_index >= arrays.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "query_index {} out of range (len={})",
                query_index,
                arrays.len()
            )));
        }
        let results = similarity::find_similar(&arrays[query_index], &arrays, threshold, top_k);
        Ok(results
            .into_iter()
            .map(|(index, score)| PySimilarityResult { index, score })
            .collect())
    }

    /// Compute cosine similarity between two bundle vectors.
    #[staticmethod]
    fn cosine_similarity(a: Vec<[f64; 2]>, b: Vec<[f64; 2]>) -> PyResult<f64> {
        let arr_a = py_vec_to_array(&a);
        let arr_b = py_vec_to_array(&b);
        Ok(similarity::cosine_similarity(&arr_a, &arr_b))
    }

    /// Return the dimensionality of the engine.
    #[getter]
    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn __repr__(&self) -> String {
        format!(
            "VsaEngine(dimensions={}, codebook_entries={})",
            self.dimensions,
            self.codebook.len()
        )
    }
}

// ---------------------------------------------------------------------------
// Standalone Python functions
// ---------------------------------------------------------------------------

/// Bundle inventory rows into hypervectors.
///
/// Convenience function that creates a temporary engine. For repeated calls,
/// prefer creating a `VsaEngine` and calling `engine.bundle_inventory()`.
#[pyfunction]
#[pyo3(signature = (rows, dimensions=1024, seed=42))]
fn bundle_inventory(
    rows: Vec<PyInventoryRow>,
    dimensions: usize,
    seed: u64,
) -> PyResult<Vec<Vec<[f64; 2]>>> {
    let primitives = VsaPrimitives::new(dimensions, seed);
    let codebook = Codebook::new(dimensions, seed);
    let rust_rows: Vec<InventoryRow> = rows.iter().map(InventoryRow::from).collect();
    let bundles = bundle_inventory_batch(&rust_rows, &primitives, &codebook);
    Ok(bundles_to_py(&bundles))
}

/// Find top-k similar vectors from a list of bundles.
#[pyfunction]
#[pyo3(signature = (query, candidates, threshold=0.0, top_k=10))]
fn find_similar_vectors(
    query: Vec<[f64; 2]>,
    candidates: Vec<Vec<[f64; 2]>>,
    threshold: f64,
    top_k: usize,
) -> PyResult<Vec<PySimilarityResult>> {
    let query_arr = py_vec_to_array(&query);
    let dim = query_arr.len();
    let cand_arrays = py_to_bundles(&candidates, dim)?;
    let results = similarity::find_similar(&query_arr, &cand_arrays, threshold, top_k);
    Ok(results
        .into_iter()
        .map(|(index, score)| PySimilarityResult { index, score })
        .collect())
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// The sentinel_vsa Python module.
#[pymodule]
pub fn sentinel_vsa(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyInventoryRow>()?;
    m.add_class::<PySimilarityResult>()?;
    m.add_class::<PyVsaEngine>()?;
    m.add_function(wrap_pyfunction!(bundle_inventory, m)?)?;
    m.add_function(wrap_pyfunction!(find_similar_vectors, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Internal conversion helpers
// ---------------------------------------------------------------------------

/// Convert a slice of Complex64 arrays to Python-friendly [real, imag] pairs.
fn bundles_to_py(bundles: &[Array1<Complex64>]) -> Vec<Vec<[f64; 2]>> {
    bundles
        .iter()
        .map(|arr| arr.iter().map(|c| [c.re, c.im]).collect())
        .collect()
}

/// Convert Python [real, imag] pairs back to Complex64 arrays.
fn py_to_bundles(
    bundles: &[Vec<[f64; 2]>],
    expected_dim: usize,
) -> PyResult<Vec<Array1<Complex64>>> {
    bundles
        .iter()
        .enumerate()
        .map(|(i, vec)| {
            if vec.len() != expected_dim {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "bundle at index {} has length {} but expected {}",
                    i,
                    vec.len(),
                    expected_dim
                )));
            }
            Ok(py_vec_to_array(vec))
        })
        .collect()
}

/// Convert a single Python [real, imag] vector to a Complex64 array.
fn py_vec_to_array(vec: &[[f64; 2]]) -> Array1<Complex64> {
    Array1::from_iter(vec.iter().map(|&[re, im]| Complex64::new(re, im)))
}
