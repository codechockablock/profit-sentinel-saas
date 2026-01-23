"""
Level 6: Column Binding Pattern Discovery

CAUSAL STRUCTURE DISCOVERY
==========================

This experiment discovers RELATIONSHIPS between columns:
- Which columns correlate?
- Which columns BIND together to predict other columns?
- What are the algebraic relationships in the data?

VSA Insight:
- Bundling (+) combines independent signals
- Binding (⊗) creates relational structure

If col_A ⊗ col_B has high similarity to col_C across many rows,
then A and B jointly determine C.

Example discovery:
  "Column 5 (Stock) ⊗ Column 0 (Sales) → Column 6 (Year Total)"
  Meaning: Stock levels and sales jointly predict yearly totals.
"""

import csv
import logging
import math
import time
from dataclasses import dataclass, field

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class ColumnRelation:
    """A discovered relationship between columns."""
    col_a: int
    col_b: int
    col_target: int
    binding_similarity: float  # How well A ⊗ B predicts target
    correlation_strength: float  # Statistical correlation
    relationship_type: str  # 'binding', 'sum', 'difference', 'ratio'
    sample_evidence: list = field(default_factory=list)


@dataclass
class ColumnProfile:
    """Statistical profile of a single column."""
    col_idx: int
    mean: float
    std: float
    min_val: float
    max_val: float
    pct_negative: float
    pct_zero: float
    distribution_type: str  # 'normal', 'skewed', 'bimodal', 'sparse'


@dataclass
class BindingDiscoveryResult:
    """Results of column binding discovery."""
    column_profiles: list[ColumnProfile]
    pairwise_correlations: dict  # (i, j) -> correlation
    discovered_bindings: list[ColumnRelation]
    discovered_formulas: list[str]  # Human-readable formulas
    discovery_time: float


# =============================================================================
# VSA COLUMN ENCODER
# =============================================================================


class ColumnBindingEncoder:
    """
    Encodes columns and discovers binding relationships.

    Key operations:
    - encode_value(v, col): Encode a single value for a column
    - bind(a, b): Create relational vector A ⊗ B
    - unbind(bound, a): Recover B from A ⊗ B given A
    """

    def __init__(self, dimensions: int = 2048, device: str | None = None):
        self.dimensions = dimensions

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.dtype = torch.complex64
        self._basis_cache = {}

    def _get_basis(self, col_idx: int) -> torch.Tensor:
        """Get random basis vector for column."""
        if col_idx not in self._basis_cache:
            gen = torch.Generator(device='cpu')
            gen.manual_seed(42 + col_idx * 1000)

            angles = torch.rand(self.dimensions, generator=gen) * 2 * torch.pi
            basis = torch.exp(1j * angles).to(self.device, self.dtype)
            self._basis_cache[col_idx] = basis

        return self._basis_cache[col_idx]

    def encode_value(self, value: float, col_idx: int) -> torch.Tensor:
        """Encode a value from a specific column."""
        basis = self._get_basis(col_idx)
        # Normalize value to reasonable phase range
        phase = torch.tanh(torch.tensor(value / 100.0)) * torch.pi
        return basis * torch.exp(1j * phase)

    def encode_column_vector(self, values: list[float], col_idx: int) -> torch.Tensor:
        """Encode all values from a column as a bundled vector."""
        bundle = torch.zeros(self.dimensions, dtype=self.dtype, device=self.device)
        for val in values:
            if val is not None and not math.isnan(val):
                bundle = bundle + self.encode_value(val, col_idx)

        norm = torch.sqrt(torch.sum(torch.abs(bundle) ** 2))
        if norm > 1e-10:
            bundle = bundle / norm
        return bundle

    def bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Binding operation: element-wise multiplication of phasors.

        In complex phasor VSA: A ⊗ B = A * B (element-wise)
        This creates a vector that represents the RELATIONSHIP between A and B.
        """
        bound = a * b
        norm = torch.sqrt(torch.sum(torch.abs(bound) ** 2))
        if norm > 1e-10:
            bound = bound / norm
        return bound

    def unbind(self, bound: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Unbinding: recover B from A ⊗ B given A.

        In complex phasor VSA: B ≈ (A ⊗ B) * conj(A)
        """
        unbound = bound * torch.conj(a)
        norm = torch.sqrt(torch.sum(torch.abs(unbound) ** 2))
        if norm > 1e-10:
            unbound = unbound / norm
        return unbound

    def similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Cosine similarity between two VSA vectors."""
        return torch.real(torch.sum(torch.conj(a) * b)).item()


# =============================================================================
# COLUMN RELATIONSHIP DISCOVERY
# =============================================================================


class ColumnRelationshipDiscoverer:
    """
    Discovers relationships between columns using VSA binding.
    """

    def __init__(self, encoder: ColumnBindingEncoder):
        self.encoder = encoder

    def profile_columns(
        self,
        data: list[list[float]],
        num_columns: int
    ) -> list[ColumnProfile]:
        """Build statistical profile for each column."""
        profiles = []

        for col_idx in range(num_columns):
            values = [row[col_idx] for row in data if col_idx < len(row) and row[col_idx] is not None]

            if not values:
                continue

            import statistics

            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0
            min_val = min(values)
            max_val = max(values)
            pct_negative = sum(1 for v in values if v < 0) / len(values) * 100
            pct_zero = sum(1 for v in values if v == 0) / len(values) * 100

            # Classify distribution
            if pct_zero > 50:
                dist_type = 'sparse'
            elif abs(mean) < std * 0.1:
                dist_type = 'symmetric'
            elif mean > 0 and min_val >= 0:
                dist_type = 'positive_skewed'
            elif pct_negative > 30:
                dist_type = 'mixed_sign'
            else:
                dist_type = 'normal'

            profiles.append(ColumnProfile(
                col_idx=col_idx,
                mean=mean,
                std=std,
                min_val=min_val,
                max_val=max_val,
                pct_negative=pct_negative,
                pct_zero=pct_zero,
                distribution_type=dist_type,
            ))

        return profiles

    def compute_pairwise_correlations(
        self,
        data: list[list[float]],
        num_columns: int,
    ) -> dict:
        """Compute statistical correlations between all column pairs."""
        correlations = {}

        for i in range(num_columns):
            for j in range(i + 1, num_columns):
                col_i = [row[i] for row in data if i < len(row) and j < len(row)]
                col_j = [row[j] for row in data if i < len(row) and j < len(row)]

                if len(col_i) < 10:
                    continue

                # Pearson correlation
                mean_i = sum(col_i) / len(col_i)
                mean_j = sum(col_j) / len(col_j)

                num = sum((a - mean_i) * (b - mean_j) for a, b in zip(col_i, col_j))
                den_i = math.sqrt(sum((a - mean_i) ** 2 for a in col_i))
                den_j = math.sqrt(sum((b - mean_j) ** 2 for b in col_j))

                if den_i > 1e-10 and den_j > 1e-10:
                    corr = num / (den_i * den_j)
                    correlations[(i, j)] = corr
                    correlations[(j, i)] = corr

        return correlations

    def discover_binding_patterns(
        self,
        data: list[list[float]],
        num_columns: int,
        top_k: int = 20,
    ) -> list[ColumnRelation]:
        """
        Discover which column pairs BIND to predict other columns.

        For each triple (A, B, C):
        - Encode values from each column per row
        - Compute A ⊗ B for each row
        - Check if A ⊗ B is similar to C across rows
        """
        logger.info("Discovering binding patterns...")

        discovered = []
        len(data)

        # Pre-encode all column values per row
        logger.info("  Encoding column values...")
        row_encodings = []
        for row_idx, row in enumerate(data):
            row_enc = {}
            for col_idx in range(min(len(row), num_columns)):
                val = row[col_idx]
                if val is not None and not math.isnan(val):
                    row_enc[col_idx] = self.encoder.encode_value(val, col_idx)
            row_encodings.append(row_enc)

        # Test binding patterns: A ⊗ B → C
        logger.info("  Testing binding patterns...")

        # Focus on most interesting columns (first 8 are usually the key metrics)
        key_columns = list(range(min(8, num_columns)))

        for col_a in key_columns:
            for col_b in key_columns:
                if col_b <= col_a:
                    continue

                for col_target in key_columns:
                    if col_target == col_a or col_target == col_b:
                        continue

                    # Compute binding similarity across all rows
                    similarities = []
                    evidence = []

                    for row_idx, row_enc in enumerate(row_encodings):
                        if col_a not in row_enc or col_b not in row_enc or col_target not in row_enc:
                            continue

                        # Bind A ⊗ B
                        bound = self.encoder.bind(row_enc[col_a], row_enc[col_b])

                        # Compare to target
                        sim = self.encoder.similarity(bound, row_enc[col_target])
                        similarities.append(sim)

                        if len(evidence) < 5 and abs(sim) > 0.5:
                            evidence.append({
                                'row': row_idx,
                                'val_a': data[row_idx][col_a],
                                'val_b': data[row_idx][col_b],
                                'val_target': data[row_idx][col_target],
                                'similarity': sim,
                            })

                    if not similarities:
                        continue

                    avg_sim = sum(similarities) / len(similarities)

                    # Also check algebraic relationships
                    rel_type, corr_strength = self._check_algebraic_relation(
                        data, col_a, col_b, col_target
                    )

                    if abs(avg_sim) > 0.3 or corr_strength > 0.7:
                        discovered.append(ColumnRelation(
                            col_a=col_a,
                            col_b=col_b,
                            col_target=col_target,
                            binding_similarity=avg_sim,
                            correlation_strength=corr_strength,
                            relationship_type=rel_type,
                            sample_evidence=evidence,
                        ))

        # Sort by strength
        discovered.sort(key=lambda x: abs(x.correlation_strength), reverse=True)

        return discovered[:top_k]

    def _check_algebraic_relation(
        self,
        data: list[list[float]],
        col_a: int,
        col_b: int,
        col_target: int,
    ) -> tuple[str, float]:
        """Check if target = A + B, A - B, A * B, or A / B."""

        best_type = 'unknown'
        best_corr = 0.0

        targets = []
        sums = []
        diffs = []
        diffs_rev = []
        products = []
        ratios = []

        for row in data:
            if col_a >= len(row) or col_b >= len(row) or col_target >= len(row):
                continue

            a, b, t = row[col_a], row[col_b], row[col_target]

            if a is None or b is None or t is None:
                continue
            if math.isnan(a) or math.isnan(b) or math.isnan(t):
                continue

            targets.append(t)
            sums.append(a + b)
            diffs.append(a - b)
            diffs_rev.append(b - a)
            products.append(a * b)
            if b != 0:
                ratios.append(a / b)
            else:
                ratios.append(0)

        if len(targets) < 10:
            return 'unknown', 0.0

        # Check each relationship type
        for name, predicted in [
            ('sum', sums),
            ('difference_ab', diffs),
            ('difference_ba', diffs_rev),
            ('product', products),
            ('ratio', ratios),
        ]:
            corr = self._pearson(targets, predicted)
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_type = name

        return best_type, best_corr

    def _pearson(self, x: list[float], y: list[float]) -> float:
        """Compute Pearson correlation."""
        n = len(x)
        if n < 2:
            return 0.0

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        num = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
        den_x = math.sqrt(sum((a - mean_x) ** 2 for a in x))
        den_y = math.sqrt(sum((b - mean_y) ** 2 for b in y))

        if den_x < 1e-10 or den_y < 1e-10:
            return 0.0

        return num / (den_x * den_y)

    def generate_formulas(
        self,
        relations: list[ColumnRelation],
        column_names: list[str],
    ) -> list[str]:
        """Generate human-readable formulas from discovered relations."""
        formulas = []

        for rel in relations:
            name_a = column_names[rel.col_a] if rel.col_a < len(column_names) else f'Col{rel.col_a}'
            name_b = column_names[rel.col_b] if rel.col_b < len(column_names) else f'Col{rel.col_b}'
            name_t = column_names[rel.col_target] if rel.col_target < len(column_names) else f'Col{rel.col_target}'

            if rel.relationship_type == 'sum':
                formula = f"{name_t} ≈ {name_a} + {name_b}"
            elif rel.relationship_type == 'difference_ab':
                formula = f"{name_t} ≈ {name_a} - {name_b}"
            elif rel.relationship_type == 'difference_ba':
                formula = f"{name_t} ≈ {name_b} - {name_a}"
            elif rel.relationship_type == 'product':
                formula = f"{name_t} ≈ {name_a} × {name_b}"
            elif rel.relationship_type == 'ratio':
                formula = f"{name_t} ≈ {name_a} / {name_b}"
            else:
                formula = f"{name_t} ~ f({name_a}, {name_b})"

            strength = f"[r={rel.correlation_strength:.3f}]"
            formulas.append(f"{formula} {strength}")

        return formulas


# =============================================================================
# DATA LOADING
# =============================================================================


def load_data_for_binding(
    csv_path: str,
    max_rows: int = 10000,
) -> tuple[list[list[float]], list[str]]:
    """Load data as raw numerical matrix with column names."""

    target_columns = [
        'Gross Sales', 'Gross Cost', 'Gross Profit', 'Profit Margin%',
        'Avg. Cost', 'Stock', 'Year Total', 'Report Total',
        'Jan', 'Last Dec', 'Last Nov', 'Last Oct', 'Last Sep',
        'Last Aug', 'Last Jul', 'Last Jun', 'Last May', 'Last Apr',
        'Last Mar', 'Last Feb', 'Last Jan'
    ]

    data = []

    with open(csv_path, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)

        # Filter to columns that exist
        available_columns = [c for c in target_columns if c in reader.fieldnames]

        for row_idx, row in enumerate(reader):
            if row_idx >= max_rows:
                break

            values = []
            for col in available_columns:
                val_str = row[col].replace(',', '').replace('$', '').replace('%', '').strip()
                try:
                    values.append(float(val_str) if val_str else 0.0)
                except ValueError:
                    values.append(0.0)

            data.append(values)

    return data, available_columns


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================


def run_level6_experiment(
    csv_path: str,
    max_rows: int = 10000,
) -> BindingDiscoveryResult:
    """Run the column binding discovery experiment."""

    print("=" * 70)
    print("LEVEL 6: COLUMN BINDING PATTERN DISCOVERY")
    print("=" * 70)
    print()
    print("Discovering RELATIONSHIPS between columns:")
    print("  - Which columns correlate?")
    print("  - Which columns BIND (⊗) to predict other columns?")
    print("  - What algebraic formulas describe the data?")
    print("=" * 70)
    print()

    # Load data
    logger.info(f"Loading data from {csv_path}...")
    data, column_names = load_data_for_binding(csv_path, max_rows)
    num_columns = len(column_names)

    print(f"Loaded {len(data)} rows × {num_columns} columns")
    print(f"Columns: {', '.join(column_names[:8])}...")
    print()

    # Create encoder and discoverer
    encoder = ColumnBindingEncoder(dimensions=2048)
    discoverer = ColumnRelationshipDiscoverer(encoder)

    start_time = time.time()

    # Profile columns
    print("-" * 70)
    print("COLUMN PROFILES")
    print("-" * 70)
    profiles = discoverer.profile_columns(data, num_columns)

    for p in profiles[:8]:  # First 8 columns
        name = column_names[p.col_idx] if p.col_idx < len(column_names) else f'Col{p.col_idx}'
        print(f"  {name:20s}: mean={p.mean:>10.2f}, std={p.std:>10.2f}, neg={p.pct_negative:>5.1f}%, zero={p.pct_zero:>5.1f}%")
    print()

    # Compute correlations
    print("-" * 70)
    print("PAIRWISE CORRELATIONS (top 10)")
    print("-" * 70)
    correlations = discoverer.compute_pairwise_correlations(data, num_columns)

    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    seen = set()
    count = 0
    for (i, j), corr in sorted_corrs:
        if (j, i) in seen:
            continue
        seen.add((i, j))

        name_i = column_names[i] if i < len(column_names) else f'Col{i}'
        name_j = column_names[j] if j < len(column_names) else f'Col{j}'
        print(f"  {name_i:20s} ↔ {name_j:20s}: r = {corr:>7.3f}")

        count += 1
        if count >= 10:
            break
    print()

    # Discover binding patterns
    print("-" * 70)
    print("DISCOVERING BINDING PATTERNS...")
    print("-" * 70)
    bindings = discoverer.discover_binding_patterns(data, num_columns, top_k=15)

    # Generate formulas
    formulas = discoverer.generate_formulas(bindings, column_names)

    discovery_time = time.time() - start_time

    result = BindingDiscoveryResult(
        column_profiles=profiles,
        pairwise_correlations=correlations,
        discovered_bindings=bindings,
        discovered_formulas=formulas,
        discovery_time=discovery_time,
    )

    return result


def print_binding_report(result: BindingDiscoveryResult, column_names: list[str]):
    """Print the binding discovery report."""

    print()
    print("=" * 70)
    print("DISCOVERED ALGEBRAIC RELATIONSHIPS")
    print("=" * 70)
    print()

    for i, formula in enumerate(result.discovered_formulas):
        print(f"  {i+1}. {formula}")

    print()
    print("-" * 70)
    print("DETAILED BINDING ANALYSIS")
    print("-" * 70)

    for rel in result.discovered_bindings[:10]:
        name_a = column_names[rel.col_a] if rel.col_a < len(column_names) else f'Col{rel.col_a}'
        name_b = column_names[rel.col_b] if rel.col_b < len(column_names) else f'Col{rel.col_b}'
        name_t = column_names[rel.col_target] if rel.col_target < len(column_names) else f'Col{rel.col_target}'

        print()
        print(f"  {name_a} ⊗ {name_b} → {name_t}")
        print(f"    VSA binding similarity: {rel.binding_similarity:.3f}")
        print(f"    Algebraic correlation:  {rel.correlation_strength:.3f}")
        print(f"    Relationship type:      {rel.relationship_type}")

        if rel.sample_evidence:
            print("    Sample evidence:")
            for ev in rel.sample_evidence[:3]:
                print(f"      Row {ev['row']}: {name_a}={ev['val_a']:.2f}, {name_b}={ev['val_b']:.2f} → {name_t}={ev['val_target']:.2f}")

    print()
    print("=" * 70)
    print(f"Discovery time: {result.discovery_time:.1f}s")
    print("=" * 70)

    # Interpretation
    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()
    print("The system discovered these relationships by:")
    print("  1. Encoding each column value as a VSA phasor vector")
    print("  2. Binding column pairs: A ⊗ B")
    print("  3. Checking if A ⊗ B predicts column C")
    print("  4. Verifying with algebraic correlation (sum, diff, product, ratio)")
    print()
    print("Strong relationships (r > 0.9) represent true data constraints.")
    print("Weaker relationships may indicate partial dependencies or anomalies.")
    print("=" * 70)


if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "/Users/joseph/Downloads/Reports/Inventory_Report_SKU_SHLP_YTD.csv"
    max_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 10000

    result = run_level6_experiment(csv_path, max_rows)

    # Get column names for report
    _, column_names = load_data_for_binding(csv_path, 1)

    print_binding_report(result, column_names)
