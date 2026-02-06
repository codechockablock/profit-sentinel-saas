"""
Level 5: Self-Derived Evaluation Criteria

PURE UNSUPERVISED STRUCTURE DISCOVERY
=====================================

This experiment strips ALL primitive definitions. The system receives:
- Raw numerical data only
- No labels, no interpretation
- No guidance on what's "bad" or "good"

The system must find structure purely from geometry:
- Cluster data geometrically
- Identify outliers
- Group rows by numerical patterns alone

Key question: Does unsupervised geometric structure discovery
produce meaningful categories, or just statistical ones?

We evaluate AFTERWARD by asking:
- Does Cluster A correspond to negative_inventory?
- Does Cluster B correspond to zero_cost_anomaly?
- Did it find something we didn't name?
- Did it find patterns that are real but semantically meaningless?
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
class DataPoint:
    """A single row of raw numerical data."""

    sku: str
    features: torch.Tensor  # Raw numerical vector
    raw_values: dict  # Original values for interpretation


@dataclass
class Cluster:
    """A discovered cluster."""

    id: int
    centroid: torch.Tensor
    members: list[int]  # Indices into data points
    radius: float  # Average distance from centroid

    # Discovered characteristics (computed post-hoc)
    feature_means: dict = field(default_factory=dict)
    feature_stds: dict = field(default_factory=dict)
    distinguishing_features: list[str] = field(default_factory=list)


@dataclass
class DiscoveryResult:
    """Results of unsupervised structure discovery."""

    clusters: list[Cluster]
    outliers: list[int]  # Indices of points not in any cluster
    feature_names: list[str]
    data_points: list[DataPoint]

    # Metrics
    silhouette_score: float = 0.0
    total_variance_explained: float = 0.0
    discovery_time: float = 0.0


# =============================================================================
# VSA ENCODER (No primitives - just numerical encoding)
# =============================================================================


class NumericalVSAEncoder:
    """
    Encodes raw numerical vectors into VSA space.

    NO PRIMITIVES. Just encodes numbers as phasor vectors.
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

        # Create random basis vectors for each feature dimension
        # These are learned/fixed, but NOT semantic primitives
        self._basis_cache = {}

    def _get_basis(self, feature_idx: int) -> torch.Tensor:
        """Get or create basis vector for a feature dimension."""
        if feature_idx not in self._basis_cache:
            # Deterministic generation from index
            gen = torch.Generator(device="cpu")
            gen.manual_seed(42 + feature_idx * 1000)

            angles = torch.rand(self.dimensions, generator=gen) * 2 * torch.pi
            basis = torch.exp(1j * angles).to(self.device, self.dtype)
            self._basis_cache[feature_idx] = basis

        return self._basis_cache[feature_idx]

    def encode_value(self, value: float, feature_idx: int) -> torch.Tensor:
        """
        Encode a single numerical value into VSA space.

        Uses phase rotation proportional to the value.
        Normalized values map to phase angles.
        """
        basis = self._get_basis(feature_idx)

        # Map value to phase rotation
        # Use tanh to bound extreme values, then scale to [-pi, pi]
        phase = torch.tanh(torch.tensor(value / 100.0)) * torch.pi

        return basis * torch.exp(1j * phase)

    def encode_vector(self, values: list[float]) -> torch.Tensor:
        """
        Encode a full numerical vector into VSA space.

        Bundle (sum) the encoded values for each feature.
        """
        bundle = torch.zeros(self.dimensions, dtype=self.dtype, device=self.device)

        for idx, val in enumerate(values):
            if val is not None and not (
                isinstance(val, float) and torch.isnan(torch.tensor(val))
            ):
                bundle = bundle + self.encode_value(val, idx)

        # Normalize
        norm = torch.sqrt(torch.sum(torch.abs(bundle) ** 2))
        if norm > 1e-10:
            bundle = bundle / norm

        return bundle

    def similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Cosine similarity between two VSA vectors."""
        return torch.real(torch.sum(torch.conj(a) * b)).item()


# =============================================================================
# GEOMETRIC CLUSTERING (No supervision)
# =============================================================================


class GeometricClusterer:
    """
    Discovers clusters purely from geometric structure.

    NO fitness function we provide.
    NO labels.
    Just finds natural groupings in the VSA space.
    """

    def __init__(
        self,
        encoder: NumericalVSAEncoder,
        min_cluster_size: int = 10,
        similarity_threshold: float = 0.3,
    ):
        self.encoder = encoder
        self.min_cluster_size = min_cluster_size
        self.similarity_threshold = similarity_threshold

    def find_clusters(
        self,
        data_points: list[DataPoint],
        max_clusters: int = 20,
    ) -> tuple[list[Cluster], list[int]]:
        """
        Find natural clusters in the data.

        Uses iterative centroid discovery:
        1. Find the point with highest average similarity to others (dense region)
        2. Assign nearby points to this cluster
        3. Remove clustered points, repeat
        4. Points not in any cluster are outliers
        """
        logger.info(f"Finding clusters in {len(data_points)} points...")

        n = len(data_points)
        assigned = [False] * n
        clusters = []

        # Stack all encodings for efficient computation
        encodings = torch.stack([dp.features for dp in data_points])

        for cluster_id in range(max_clusters):
            # Find unassigned points
            unassigned_idx = [i for i in range(n) if not assigned[i]]
            if len(unassigned_idx) < self.min_cluster_size:
                break

            # Compute similarity matrix for unassigned points
            unassigned_encodings = encodings[unassigned_idx]

            # Find densest point (highest average similarity to others)
            sims = torch.real(torch.conj(unassigned_encodings) @ unassigned_encodings.T)
            avg_sims = sims.mean(dim=1)

            # The densest point becomes the seed
            seed_local_idx = avg_sims.argmax().item()
            seed_global_idx = unassigned_idx[seed_local_idx]
            seed_encoding = encodings[seed_global_idx]

            # Find all points similar to seed
            all_sims = torch.real(torch.conj(seed_encoding) @ encodings.T)

            # Assign points above threshold that aren't already assigned
            members = []
            for i in range(n):
                if not assigned[i] and all_sims[i].item() > self.similarity_threshold:
                    members.append(i)
                    assigned[i] = True

            if len(members) < self.min_cluster_size:
                # Unassign and mark as potential outliers
                for m in members:
                    assigned[m] = False
                continue

            # Compute cluster centroid
            member_encodings = encodings[members]
            centroid = member_encodings.mean(dim=0)
            centroid = centroid / torch.sqrt(torch.sum(torch.abs(centroid) ** 2))

            # Compute radius (average distance from centroid)
            centroid_sims = torch.real(torch.conj(centroid) @ member_encodings.T)
            radius = 1 - centroid_sims.mean().item()

            cluster = Cluster(
                id=cluster_id,
                centroid=centroid,
                members=members,
                radius=radius,
            )
            clusters.append(cluster)

            logger.info(
                f"  Cluster {cluster_id}: {len(members)} members, radius={radius:.3f}"
            )

        # Remaining unassigned points are outliers
        outliers = [i for i in range(n) if not assigned[i]]

        logger.info(f"Found {len(clusters)} clusters, {len(outliers)} outliers")

        return clusters, outliers

    def characterize_clusters(
        self,
        clusters: list[Cluster],
        data_points: list[DataPoint],
        feature_names: list[str],
    ) -> None:
        """
        Compute distinguishing characteristics of each cluster.

        This is POST-HOC analysis - the clusters were found without this.
        """
        logger.info("Characterizing clusters...")

        # Compute global stats
        all_values = {name: [] for name in feature_names}
        for dp in data_points:
            for name in feature_names:
                val = dp.raw_values.get(name)
                if val is not None:
                    all_values[name].append(val)

        global_means = {
            name: sum(vals) / len(vals) if vals else 0
            for name, vals in all_values.items()
        }
        global_stds = {
            name: (
                (sum((v - global_means[name]) ** 2 for v in vals) / len(vals)) ** 0.5
                if vals
                else 1
            )
            for name, vals in all_values.items()
        }

        for cluster in clusters:
            # Compute cluster stats
            cluster_values = {name: [] for name in feature_names}
            for idx in cluster.members:
                dp = data_points[idx]
                for name in feature_names:
                    val = dp.raw_values.get(name)
                    if val is not None:
                        cluster_values[name].append(val)

            cluster.feature_means = {
                name: sum(vals) / len(vals) if vals else 0
                for name, vals in cluster_values.items()
            }
            cluster.feature_stds = {
                name: (
                    (
                        sum((v - cluster.feature_means[name]) ** 2 for v in vals)
                        / len(vals)
                    )
                    ** 0.5
                    if vals
                    else 0
                )
                for name, vals in cluster_values.items()
            }

            # Find distinguishing features (deviation from global mean)
            deviations = []
            for name in feature_names:
                if global_stds[name] > 0.01:  # Avoid division by tiny numbers
                    z_score = (
                        cluster.feature_means[name] - global_means[name]
                    ) / global_stds[name]
                    deviations.append((name, z_score, cluster.feature_means[name]))

            # Sort by absolute z-score
            deviations.sort(key=lambda x: abs(x[1]), reverse=True)

            cluster.distinguishing_features = [
                f"{name}: {mean:.2f} (z={z:.2f})" for name, z, mean in deviations[:5]
            ]


# =============================================================================
# DATA LOADING
# =============================================================================


def load_ytd_numerical_data(
    csv_path: str,
    max_rows: int = 10000,
) -> tuple[list[DataPoint], list[str]]:
    """
    Load ONLY numerical columns from YTD data.

    No labels, no interpretation.
    """
    logger.info(f"Loading numerical data from {csv_path}...")

    # Define which columns are numerical
    numerical_columns = [
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

    encoder = NumericalVSAEncoder()
    data_points = []

    with open(csv_path, encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)

        # Find which columns actually exist
        available_columns = []
        for row in reader:
            for col in numerical_columns:
                if col in row:
                    available_columns.append(col)
            break

        # Reset and read
        f.seek(0)
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            if i >= max_rows:
                break

            try:
                sku = row.get("SKU", f"row_{i}")

                # Extract numerical values
                values = []
                raw_values = {}

                for col in available_columns:
                    val_str = row.get(col, "")
                    if val_str and val_str.strip():
                        try:
                            val = float(
                                val_str.replace(",", "")
                                .replace("$", "")
                                .replace("%", "")
                            )
                            values.append(val)
                            raw_values[col] = val
                        except ValueError:
                            values.append(0.0)
                            raw_values[col] = 0.0
                    else:
                        values.append(0.0)
                        raw_values[col] = 0.0

                # Encode into VSA space
                encoding = encoder.encode_vector(values)

                data_points.append(
                    DataPoint(
                        sku=sku,
                        features=encoding,
                        raw_values=raw_values,
                    )
                )

            except Exception:
                continue

        if i > 0 and i % 2000 == 0:
            logger.info(f"  Loaded {i} rows...")

    logger.info(
        f"Loaded {len(data_points)} data points with {len(available_columns)} features"
    )

    return data_points, available_columns


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================


def run_level5_experiment(
    csv_path: str,
    max_rows: int = 10000,
    output_path: str | None = None,
) -> DiscoveryResult:
    """
    Run the Level 5 self-grounding experiment.

    Pure unsupervised structure discovery.
    """
    start_time = time.time()

    print("=" * 70)
    print("LEVEL 5: SELF-DERIVED EVALUATION CRITERIA")
    print("=" * 70)
    print("\nPure unsupervised structure discovery.")
    print("NO primitives. NO labels. NO guidance.")
    print("The system finds structure from geometry alone.")
    print("=" * 70)

    # Load data
    data_points, feature_names = load_ytd_numerical_data(csv_path, max_rows)

    print(f"\nLoaded {len(data_points)} data points")
    print(f"Features: {feature_names}")

    # Create encoder and clusterer
    encoder = NumericalVSAEncoder()
    clusterer = GeometricClusterer(
        encoder=encoder,
        min_cluster_size=max(10, len(data_points) // 100),
        similarity_threshold=0.25,
    )

    # Find clusters
    print("\n" + "-" * 70)
    print("DISCOVERING STRUCTURE...")
    print("-" * 70)

    clusters, outliers = clusterer.find_clusters(data_points)

    # Characterize clusters
    clusterer.characterize_clusters(clusters, data_points, feature_names)

    discovery_time = time.time() - start_time

    # Create result
    result = DiscoveryResult(
        clusters=clusters,
        outliers=outliers,
        feature_names=feature_names,
        data_points=data_points,
        discovery_time=discovery_time,
    )

    # Print results
    print_discovery_report(result)

    # Save if requested
    if output_path:
        save_discovery_report(result, output_path)

    return result


def print_discovery_report(result: DiscoveryResult) -> None:
    """Print the discovery report."""
    print("\n" + "=" * 70)
    print("DISCOVERY RESULTS")
    print("=" * 70)

    print(f"\nClusters found: {len(result.clusters)}")
    print(f"Outliers: {len(result.outliers)}")
    print(f"Discovery time: {result.discovery_time:.1f}s")

    for cluster in result.clusters:
        print("\n" + "-" * 70)
        print(f"CLUSTER {cluster.id}")
        print("-" * 70)
        print(f"  Members: {len(cluster.members)}")
        print(f"  Radius: {cluster.radius:.3f}")

        print("\n  DISTINGUISHING FEATURES:")
        for feat in cluster.distinguishing_features:
            print(f"    {feat}")

        print("\n  EXAMPLE ROWS (first 10):")
        for i, idx in enumerate(cluster.members[:10]):
            dp = result.data_points[idx]
            # Show key features
            stock = dp.raw_values.get("Stock", 0)
            sales = dp.raw_values.get("Gross Sales", 0)
            profit = dp.raw_values.get("Gross Profit", 0)
            margin = dp.raw_values.get("Profit Margin%", 0)
            cost = dp.raw_values.get("Gross Cost", 0)

            print(
                f"    {i+1}. {dp.sku[:25]:<25} | Stock={stock:>8.0f} | Sales=${sales:>10.2f} | Profit=${profit:>10.2f} | Margin={margin:>6.1f}% | Cost=${cost:>10.2f}"
            )

    if result.outliers:
        print("\n" + "-" * 70)
        print(f"OUTLIERS ({len(result.outliers)} total)")
        print("-" * 70)
        print("  Sample outliers (first 10):")
        for i, idx in enumerate(result.outliers[:10]):
            dp = result.data_points[idx]
            stock = dp.raw_values.get("Stock", 0)
            sales = dp.raw_values.get("Gross Sales", 0)
            profit = dp.raw_values.get("Gross Profit", 0)
            margin = dp.raw_values.get("Profit Margin%", 0)

            print(
                f"    {i+1}. {dp.sku[:25]:<25} | Stock={stock:>8.0f} | Sales=${sales:>10.2f} | Profit=${profit:>10.2f} | Margin={margin:>6.1f}%"
            )

    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE (for human analysis)")
    print("=" * 70)
    print(
        """
After reviewing clusters, ask:

1. Does any cluster correspond to 'negative_inventory'?
   Look for: Stock < 0, negative values in Stock column

2. Does any cluster correspond to 'zero_cost_anomaly'?
   Look for: Gross Cost = 0 but Gross Sales > 0

3. Does any cluster correspond to 'negative_profit'?
   Look for: Gross Profit < 0

4. Does any cluster correspond to 'dead_item'?
   Look for: Stock > 0 but Year Total = 0 or very low

5. Did it find something we didn't name?
   Look for: Unexpected feature combinations

6. Are any clusters statistically real but semantically meaningless?
   Look for: Clusters distinguished only by arbitrary value ranges
"""
    )
    print("=" * 70)


def save_discovery_report(result: DiscoveryResult, output_path: str) -> None:
    """Save the discovery report to file."""
    import io
    from contextlib import redirect_stdout

    f = io.StringIO()
    with redirect_stdout(f):
        print_discovery_report(result)

    with open(output_path, "w") as out:
        out.write(f.getvalue())

        # Add raw cluster data
        out.write("\n\n" + "=" * 70 + "\n")
        out.write("RAW CLUSTER DATA\n")
        out.write("=" * 70 + "\n")

        for cluster in result.clusters:
            out.write(f"\nCluster {cluster.id}:\n")
            out.write(f"  Feature means: {cluster.feature_means}\n")
            out.write(f"  Feature stds: {cluster.feature_stds}\n")
            out.write(f"  All members ({len(cluster.members)}):\n")
            for idx in cluster.members:
                dp = result.data_points[idx]
                out.write(f"    {dp.sku}: {dp.raw_values}\n")

    print(f"\nReport saved to: {output_path}")


# =============================================================================
# ENTRY POINT
# =============================================================================


if __name__ == "__main__":
    import sys
    from datetime import datetime

    csv_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "/Users/joseph/Downloads/Reports/Inventory_Report_SKU_SHLP_YTD.csv"
    )
    max_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 10000

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"/Users/joseph/profit-sentinel-saas/packages/vsa-core/sandbox/level5_results_{timestamp}.txt"

    result = run_level5_experiment(
        csv_path=csv_path,
        max_rows=max_rows,
        output_path=output_path,
    )
