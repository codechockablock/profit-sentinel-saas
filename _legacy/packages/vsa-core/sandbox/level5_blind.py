"""
Level 5 BLIND: Zero Semantic Information

TRULY UNSUPERVISED STRUCTURE DISCOVERY
======================================

This experiment strips EVERYTHING:
- No column names (just Column_0, Column_1, ...)
- No SKU names (just Row_0, Row_1, ...)
- No labels, no interpretation
- Pure numbers only

The question: Does structure exist in the NUMBERS THEMSELVES,
or did we accidentally leak it through column names?

If the blind version finds similar clusters to the named version,
then the structure is in the data geometry, not the semantics.
"""

import csv
import logging
import time
from dataclasses import dataclass, field

import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class BlindDataPoint:
    """A single row - NO identifying information."""

    row_id: int  # Just a number, not SKU
    features: torch.Tensor
    raw_values: list[float]  # Just numbers, no column names


@dataclass
class BlindCluster:
    """A discovered cluster - no semantic labels."""

    id: int
    centroid: torch.Tensor
    members: list[int]
    radius: float

    # Statistics per column (not named)
    column_means: list[float] = field(default_factory=list)
    column_stds: list[float] = field(default_factory=list)


@dataclass
class BlindDiscoveryResult:
    """Results with zero semantic information."""

    clusters: list[BlindCluster]
    outliers: list[int]
    num_columns: int
    data_points: list[BlindDataPoint]
    discovery_time: float = 0.0


# =============================================================================
# BLIND VSA ENCODER
# =============================================================================


class BlindVSAEncoder:
    """
    Encodes raw numerical vectors into VSA space.

    ZERO semantic information:
    - Columns are just indices (0, 1, 2, ...)
    - Values are just numbers
    - No interpretation whatsoever
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
        """Get random basis vector for column index."""
        if col_idx not in self._basis_cache:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(42 + col_idx * 1000)

            angles = torch.rand(self.dimensions, generator=gen) * 2 * torch.pi
            basis = torch.exp(1j * angles).to(self.device, self.dtype)
            self._basis_cache[col_idx] = basis

        return self._basis_cache[col_idx]

    def encode_value(self, value: float, col_idx: int) -> torch.Tensor:
        """Encode a number. Just a number."""
        basis = self._get_basis(col_idx)
        phase = torch.tanh(torch.tensor(value / 100.0)) * torch.pi
        return basis * torch.exp(1j * phase)

    def encode_vector(self, values: list[float]) -> torch.Tensor:
        """Encode a row of numbers."""
        bundle = torch.zeros(self.dimensions, dtype=self.dtype, device=self.device)

        for idx, val in enumerate(values):
            if val is not None and not (
                isinstance(val, float) and torch.isnan(torch.tensor(val))
            ):
                bundle = bundle + self.encode_value(val, idx)

        norm = torch.sqrt(torch.sum(torch.abs(bundle) ** 2))
        if norm > 1e-10:
            bundle = bundle / norm

        return bundle

    def similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Cosine similarity."""
        return torch.real(torch.sum(torch.conj(a) * b)).item()


# =============================================================================
# BLIND CLUSTERING
# =============================================================================


class BlindClusterer:
    """Finds clusters with zero semantic guidance."""

    def __init__(
        self,
        encoder: BlindVSAEncoder,
        min_cluster_size: int = 50,
        similarity_threshold: float = 0.7,
    ):
        self.encoder = encoder
        self.min_cluster_size = min_cluster_size
        self.similarity_threshold = similarity_threshold

    def find_clusters(
        self,
        data_points: list[BlindDataPoint],
        max_clusters: int = 20,
    ) -> tuple[list[BlindCluster], list[int]]:
        """Find clusters from geometry alone."""
        logger.info(f"Finding clusters in {len(data_points)} points...")

        n = len(data_points)
        assigned = [False] * n
        clusters = []

        encodings = torch.stack([dp.features for dp in data_points])

        for cluster_id in range(max_clusters):
            unassigned_idx = [i for i in range(n) if not assigned[i]]
            if len(unassigned_idx) < self.min_cluster_size:
                break

            unassigned_encodings = encodings[unassigned_idx]

            sims = torch.real(torch.conj(unassigned_encodings) @ unassigned_encodings.T)
            avg_sims = sims.mean(dim=1)

            seed_local_idx = avg_sims.argmax().item()
            seed_global_idx = unassigned_idx[seed_local_idx]
            seed_encoding = encodings[seed_global_idx]

            all_sims = torch.real(torch.conj(seed_encoding) @ encodings.T)

            members = []
            for i in range(n):
                if not assigned[i] and all_sims[i].item() > self.similarity_threshold:
                    members.append(i)
                    assigned[i] = True

            if len(members) < self.min_cluster_size:
                for m in members:
                    assigned[m] = False
                continue

            member_encodings = encodings[members]
            centroid = member_encodings.mean(dim=0)
            centroid = centroid / torch.sqrt(torch.sum(torch.abs(centroid) ** 2))

            distances = 1 - torch.real(torch.conj(centroid) @ member_encodings.T)
            radius = distances.mean().item()

            cluster = BlindCluster(
                id=cluster_id,
                centroid=centroid,
                members=members,
                radius=radius,
            )

            clusters.append(cluster)
            logger.info(
                f"  Cluster {cluster_id}: {len(members)} members, radius={radius:.3f}"
            )

        outliers = [i for i in range(n) if not assigned[i]]
        logger.info(f"Found {len(clusters)} clusters, {len(outliers)} outliers")

        return clusters, outliers


# =============================================================================
# BLIND DATA LOADER
# =============================================================================


def load_blind_data(
    csv_path: str,
    max_rows: int = 10000,
) -> tuple[list[BlindDataPoint], int]:
    """
    Load data with ZERO semantic information.

    - Column names are discarded
    - SKU names are discarded
    - Just returns rows of numbers
    """
    logger.info(f"Loading BLIND data from {csv_path}...")

    # Numerical columns we extract (by position after we find them)
    target_columns = [
        "Gross Sales",
        "Gross Cost",
        "Gross Profit",
        "Profit Margin%",
        "Avg. Cost",
        "Stock",
        "Year Total",
        "Report Total",
        "Jan",
        "Last Dec",
        "Last Nov",
        "Last Oct",
        "Last Sep",
        "Last Aug",
        "Last Jul",
        "Last Jun",
        "Last May",
        "Last Apr",
        "Last Mar",
        "Last Feb",
        "Last Jan",
    ]

    encoder = BlindVSAEncoder(dimensions=2048)
    data_points = []

    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        # Find column indices (then FORGET the names)
        fieldnames = reader.fieldnames
        col_indices = []
        for col in target_columns:
            if col in fieldnames:
                col_indices.append(fieldnames.index(col))

        num_columns = len(col_indices)

        for row_id, row in enumerate(reader):
            if row_id >= max_rows:
                break

            # Extract just the numbers - NO NAMES
            values = []
            for col in target_columns:
                if col in row:
                    val_str = (
                        row[col]
                        .replace(",", "")
                        .replace("$", "")
                        .replace("%", "")
                        .strip()
                    )
                    try:
                        values.append(float(val_str) if val_str else 0.0)
                    except ValueError:
                        values.append(0.0)

            if not values:
                continue

            # Encode with ZERO semantic info
            encoding = encoder.encode_vector(values)

            dp = BlindDataPoint(
                row_id=row_id,  # Just a number
                features=encoding,
                raw_values=values,  # Just numbers, no names
            )
            data_points.append(dp)

            if len(data_points) % 10000 == 0:
                logger.info(f"  Loaded {len(data_points)} rows...")

    logger.info(
        f"Loaded {len(data_points)} data points with {num_columns} columns (unnamed)"
    )
    return data_points, num_columns


# =============================================================================
# BLIND ANALYSIS
# =============================================================================


def analyze_blind_results(
    result: BlindDiscoveryResult,
    original_csv_path: str,
) -> dict:
    """
    After blind clustering, we peek at the original data
    to see what the clusters actually contain.

    This is the EVALUATION step - separate from discovery.
    """
    # Load original data WITH names for comparison
    sku_map = {}
    column_names = [
        "Gross Sales",
        "Gross Cost",
        "Gross Profit",
        "Profit Margin%",
        "Avg. Cost",
        "Stock",
        "Year Total",
        "Report Total",
        "Jan",
        "Last Dec",
        "Last Nov",
        "Last Oct",
        "Last Sep",
        "Last Aug",
        "Last Jul",
        "Last Jun",
        "Last May",
        "Last Apr",
        "Last Mar",
        "Last Feb",
        "Last Jan",
    ]

    with open(original_csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row_id, row in enumerate(reader):
            if row_id >= len(result.data_points):
                break
            sku_map[row_id] = row.get("SKU #", row.get("SKU", f"Row_{row_id}"))

    analysis = {
        "clusters": [],
        "outliers": [],
    }

    # Analyze each cluster
    for cluster in result.clusters:
        cluster_analysis = {
            "id": cluster.id,
            "size": len(cluster.members),
            "radius": cluster.radius,
            "column_stats": {},
            "anomaly_counts": {},
            "sample_skus": [],
        }

        # Compute stats per column
        for col_idx in range(result.num_columns):
            values = [
                result.data_points[m].raw_values[col_idx] for m in cluster.members
            ]
            col_name = (
                column_names[col_idx]
                if col_idx < len(column_names)
                else f"Col_{col_idx}"
            )

            import statistics

            cluster_analysis["column_stats"][col_name] = {
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0,
            }

        # Count anomalies (using column positions, not names)
        # Column 5 = Stock, Column 1 = Gross Cost, Column 0 = Gross Sales, Column 2 = Gross Profit
        stocks = [result.data_points[m].raw_values[5] for m in cluster.members]
        costs = [result.data_points[m].raw_values[1] for m in cluster.members]
        sales = [result.data_points[m].raw_values[0] for m in cluster.members]
        profits = [result.data_points[m].raw_values[2] for m in cluster.members]

        cluster_analysis["anomaly_counts"] = {
            "negative_col5": sum(1 for s in stocks if s < 0),
            "zero_col1_with_col0": sum(
                1 for i, c in enumerate(costs) if c == 0 and sales[i] > 0
            ),
            "negative_col2": sum(1 for p in profits if p < 0),
        }

        # Sample SKUs
        for m in cluster.members[:10]:
            cluster_analysis["sample_skus"].append(
                {
                    "row_id": m,
                    "sku": sku_map.get(m, f"Row_{m}"),
                    "values": result.data_points[m].raw_values[:6],  # First 6 columns
                }
            )

        analysis["clusters"].append(cluster_analysis)

    # Analyze outliers
    for m in result.outliers[:20]:
        analysis["outliers"].append(
            {
                "row_id": m,
                "sku": sku_map.get(m, f"Row_{m}"),
                "values": result.data_points[m].raw_values[:6],
            }
        )

    return analysis


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================


def run_blind_experiment(
    csv_path: str,
    max_rows: int = 10000,
    similarity_threshold: float = 0.7,
    min_cluster_size: int = 50,
) -> BlindDiscoveryResult:
    """Run the blind structure discovery experiment."""

    print("=" * 70)
    print("LEVEL 5 BLIND: ZERO SEMANTIC INFORMATION")
    print("=" * 70)
    print()
    print("The system receives:")
    print("  - Rows of numbers (no SKU names)")
    print("  - Columns of numbers (no column names)")
    print("  - NOTHING ELSE")
    print()
    print("Question: Is structure in the NUMBERS or the NAMES?")
    print("=" * 70)
    print()

    # Load blind data
    data_points, num_columns = load_blind_data(csv_path, max_rows)
    print(f"Loaded {len(data_points)} rows x {num_columns} columns (all unnamed)")
    print()

    # Create encoder and clusterer
    encoder = BlindVSAEncoder(dimensions=2048)
    clusterer = BlindClusterer(
        encoder=encoder,
        min_cluster_size=min_cluster_size,
        similarity_threshold=similarity_threshold,
    )

    # Find clusters
    print("-" * 70)
    print("DISCOVERING STRUCTURE (BLIND)...")
    print("-" * 70)
    print()

    start_time = time.time()
    clusters, outliers = clusterer.find_clusters(data_points, max_clusters=20)
    discovery_time = time.time() - start_time

    result = BlindDiscoveryResult(
        clusters=clusters,
        outliers=outliers,
        num_columns=num_columns,
        data_points=data_points,
        discovery_time=discovery_time,
    )

    return result


def print_blind_report(result: BlindDiscoveryResult, analysis: dict):
    """Print the blind discovery report."""

    print()
    print("=" * 70)
    print("BLIND DISCOVERY RESULTS")
    print("=" * 70)
    print()
    print(f"Clusters found: {len(result.clusters)}")
    print(f"Outliers: {len(result.outliers)}")
    print(f"Discovery time: {result.discovery_time:.1f}s")
    print()

    column_names = [
        "Gross Sales",
        "Gross Cost",
        "Gross Profit",
        "Profit Margin%",
        "Avg. Cost",
        "Stock",
    ]

    for ca in analysis["clusters"]:
        print("-" * 70)
        print(f"CLUSTER {ca['id']}")
        print("-" * 70)
        print(f"  Members: {ca['size']}")
        print(f"  Radius: {ca['radius']:.3f}")
        print()

        print("  COLUMN STATISTICS (revealed post-hoc):")
        for col_name in column_names:
            if col_name in ca["column_stats"]:
                stats = ca["column_stats"][col_name]
                print(
                    f"    {col_name:20s}: min={stats['min']:>10.2f}, max={stats['max']:>10.2f}, mean={stats['mean']:>10.2f}"
                )
        print()

        print("  ANOMALY COUNTS (revealed post-hoc):")
        print(
            f"    negative_stock:     {ca['anomaly_counts']['negative_col5']} ({100*ca['anomaly_counts']['negative_col5']/ca['size']:.1f}%)"
        )
        print(
            f"    zero_cost_w_sales:  {ca['anomaly_counts']['zero_col1_with_col0']} ({100*ca['anomaly_counts']['zero_col1_with_col0']/ca['size']:.1f}%)"
        )
        print(
            f"    negative_profit:    {ca['anomaly_counts']['negative_col2']} ({100*ca['anomaly_counts']['negative_col2']/ca['size']:.1f}%)"
        )
        print()

        print("  SAMPLE ROWS (revealed post-hoc):")
        for sample in ca["sample_skus"][:5]:
            sku = sample["sku"][:25].ljust(25)
            vals = sample["values"]
            print(
                f"    {sku} | Col0=${vals[0]:>10.2f} | Col1=${vals[1]:>10.2f} | Col2=${vals[2]:>8.2f} | Col5={vals[5]:>7.0f}"
            )
        print()

    print("-" * 70)
    print(f"OUTLIERS ({len(result.outliers)} total, showing first 10)")
    print("-" * 70)
    for out in analysis["outliers"][:10]:
        sku = out["sku"][:25].ljust(25)
        vals = out["values"]
        print(
            f"  {sku} | Col0=${vals[0]:>10.2f} | Col1=${vals[1]:>10.2f} | Col2=${vals[2]:>8.2f} | Col5={vals[5]:>7.0f}"
        )
    print()

    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()
    print("The system found these clusters using ONLY numerical patterns.")
    print("Column names and SKU names were hidden during discovery.")
    print()
    print("If clusters match the named version, structure is in the NUMBERS.")
    print("If clusters differ, we were leaking semantics through names.")
    print("=" * 70)


if __name__ == "__main__":
    import sys

    csv_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "/Users/joseph/Downloads/Reports/Inventory_Report_SKU_SHLP_YTD.csv"
    )
    max_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 10000

    result = run_blind_experiment(csv_path, max_rows)
    analysis = analyze_blind_results(result, csv_path)
    print_blind_report(result, analysis)
