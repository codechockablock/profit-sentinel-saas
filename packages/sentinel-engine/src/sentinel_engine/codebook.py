"""
engine/codebook.py - Persistent Codebook Management

Provides efficient codebook storage and retrieval for large-scale VSA:
- Memory-mapped storage for codebooks
- LRU caching with actual access tracking
- Session-based sharding for multi-tenant use
- Incremental updates and versioning
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from collections import OrderedDict
import time
import json
import hashlib
import logging
import threading

import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CodebookMetadata:
    """Metadata for a codebook."""
    name: str
    domain: str
    dimensions: int
    vector_count: int
    created_at: float
    updated_at: float
    version: str
    checksum: str
    dtype: str = "complex64"


class PersistentCodebook:
    """Memory-mapped persistent codebook.

    Stores vectors on disk with memory mapping for efficient
    random access without loading entire codebook into RAM.

    Example:
        codebook = PersistentCodebook("./codebooks/retail")
        codebook.add("SKU123", vector)
        codebook.save()

        # Later...
        codebook = PersistentCodebook("./codebooks/retail")
        codebook.load()
        vec = codebook.get("SKU123")
    """

    def __init__(
        self,
        path: str,
        dimensions: int = 16384,
        max_size: int = 100000
    ):
        """Initialize codebook.

        Args:
            path: Directory path for storage
            dimensions: Vector dimensionality
            max_size: Maximum number of vectors
        """
        self.path = Path(path)
        self.dimensions = dimensions
        self.max_size = max_size

        self._labels: List[str] = []
        self._label_to_idx: Dict[str, int] = {}
        self._vectors: Optional[torch.Tensor] = None
        self._metadata: Optional[CodebookMetadata] = None
        self._dirty = False

        # Memory map file
        self._mmap: Optional[np.memmap] = None

        # Lock for thread safety
        self._lock = threading.RLock()

    def _ensure_dir(self) -> None:
        """Ensure storage directory exists."""
        self.path.mkdir(parents=True, exist_ok=True)

    def add(self, label: str, vector: torch.Tensor) -> int:
        """Add vector to codebook.

        Args:
            label: Label for the vector
            vector: Vector to add

        Returns:
            Index of added vector
        """
        with self._lock:
            if label in self._label_to_idx:
                # Update existing
                idx = self._label_to_idx[label]
                self._vectors[idx] = vector
            else:
                # Add new
                if len(self._labels) >= self.max_size:
                    raise ValueError(f"Codebook full ({self.max_size} vectors)")

                idx = len(self._labels)
                self._labels.append(label)
                self._label_to_idx[label] = idx

                if self._vectors is None:
                    self._vectors = vector.unsqueeze(0)
                else:
                    self._vectors = torch.cat([
                        self._vectors,
                        vector.unsqueeze(0)
                    ], dim=0)

            self._dirty = True
            return idx

    def get(self, label: str) -> Optional[torch.Tensor]:
        """Get vector by label."""
        with self._lock:
            idx = self._label_to_idx.get(label)
            if idx is None:
                return None
            return self._vectors[idx]

    def get_batch(self, labels: List[str]) -> Tuple[List[str], torch.Tensor]:
        """Get multiple vectors by labels.

        Args:
            labels: Labels to retrieve

        Returns:
            (found_labels, vectors) tuple
        """
        with self._lock:
            found = []
            vectors = []

            for label in labels:
                idx = self._label_to_idx.get(label)
                if idx is not None:
                    found.append(label)
                    vectors.append(self._vectors[idx])

            if not vectors:
                return [], torch.empty(0)

            return found, torch.stack(vectors)

    def remove(self, label: str) -> bool:
        """Remove vector from codebook.

        Note: This marks for removal; actual compaction happens on save.
        """
        with self._lock:
            if label not in self._label_to_idx:
                return False

            idx = self._label_to_idx.pop(label)
            self._labels[idx] = None  # Mark as removed
            self._dirty = True
            return True

    def save(self) -> None:
        """Save codebook to disk."""
        with self._lock:
            self._ensure_dir()

            # Compact (remove None entries)
            self._compact()

            if self._vectors is None or len(self._labels) == 0:
                return

            # Save vectors as numpy (memory-mappable)
            vectors_np = torch.view_as_real(self._vectors).cpu().numpy()
            vectors_path = self.path / "vectors.npy"
            np.save(vectors_path, vectors_np)

            # Save labels
            labels_path = self.path / "labels.json"
            with open(labels_path, 'w') as f:
                json.dump(self._labels, f)

            # Compute checksum
            checksum = hashlib.sha256(vectors_np.tobytes()).hexdigest()[:16]

            # Save metadata
            self._metadata = CodebookMetadata(
                name=self.path.name,
                domain="",
                dimensions=self.dimensions,
                vector_count=len(self._labels),
                created_at=self._metadata.created_at if self._metadata else time.time(),
                updated_at=time.time(),
                version="1.0.0",
                checksum=checksum
            )

            meta_path = self.path / "metadata.json"
            with open(meta_path, 'w') as f:
                json.dump(self._metadata.__dict__, f, indent=2)

            self._dirty = False
            logger.info(f"Saved codebook: {len(self._labels)} vectors to {self.path}")

    def load(self, mmap: bool = True) -> None:
        """Load codebook from disk.

        Args:
            mmap: Use memory mapping for vectors
        """
        with self._lock:
            vectors_path = self.path / "vectors.npy"
            labels_path = self.path / "labels.json"
            meta_path = self.path / "metadata.json"

            if not vectors_path.exists():
                logger.warning(f"No codebook found at {self.path}")
                return

            # Load labels
            with open(labels_path) as f:
                self._labels = json.load(f)

            self._label_to_idx = {l: i for i, l in enumerate(self._labels)}

            # Load vectors
            if mmap:
                self._mmap = np.load(vectors_path, mmap_mode='r')
                vectors_np = self._mmap
            else:
                vectors_np = np.load(vectors_path)

            # Convert back to complex
            vectors_real = torch.from_numpy(vectors_np.copy())
            # Reshape from (n, d, 2) back to (n, d) complex
            self._vectors = torch.view_as_complex(vectors_real)

            # Load metadata
            if meta_path.exists():
                with open(meta_path) as f:
                    self._metadata = CodebookMetadata(**json.load(f))

            self._dirty = False
            logger.info(f"Loaded codebook: {len(self._labels)} vectors from {self.path}")

    def _compact(self) -> None:
        """Remove None entries from labels and vectors."""
        if not any(l is None for l in self._labels):
            return

        new_labels = []
        new_vectors = []

        for i, label in enumerate(self._labels):
            if label is not None:
                new_labels.append(label)
                new_vectors.append(self._vectors[i])

        self._labels = new_labels
        self._label_to_idx = {l: i for i, l in enumerate(self._labels)}
        self._vectors = torch.stack(new_vectors) if new_vectors else None

    def __len__(self) -> int:
        return len(self._labels)

    def __contains__(self, label: str) -> bool:
        return label in self._label_to_idx

    @property
    def labels(self) -> List[str]:
        return list(self._labels)

    @property
    def vectors(self) -> Optional[torch.Tensor]:
        return self._vectors


class LRUCodebook:
    """LRU-cached codebook with true access tracking.

    Unlike simple FIFO, this tracks actual access patterns
    and evicts least recently used vectors.
    """

    def __init__(self, max_size: int = 50000, dimensions: int = 16384):
        self.max_size = max_size
        self.dimensions = dimensions

        # OrderedDict for LRU ordering
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._access_count: Dict[str, int] = {}
        self._lock = threading.RLock()

    def get(self, label: str) -> Optional[torch.Tensor]:
        """Get vector, updating access tracking."""
        with self._lock:
            if label not in self._cache:
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(label)
            self._access_count[label] = self._access_count.get(label, 0) + 1
            return self._cache[label]

    def put(self, label: str, vector: torch.Tensor) -> None:
        """Add vector, evicting LRU if needed."""
        with self._lock:
            if label in self._cache:
                self._cache.move_to_end(label)
                self._cache[label] = vector
            else:
                # Evict if at capacity
                while len(self._cache) >= self.max_size:
                    evicted_label, _ = self._cache.popitem(last=False)
                    self._access_count.pop(evicted_label, None)

                self._cache[label] = vector
                self._access_count[label] = 0

    def evict(self, label: str) -> bool:
        """Manually evict a vector."""
        with self._lock:
            if label in self._cache:
                del self._cache[label]
                self._access_count.pop(label, None)
                return True
            return False

    def clear(self) -> None:
        """Clear all cached vectors."""
        with self._lock:
            self._cache.clear()
            self._access_count.clear()

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, label: str) -> bool:
        return label in self._cache

    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_accesses = sum(self._access_count.values())
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "total_accesses": total_accesses,
                "avg_accesses": total_accesses / max(1, len(self._cache)),
            }


class CodebookManager:
    """Manages multiple codebooks with session isolation.

    Provides multi-tenant codebook management with:
    - Session-based isolation
    - Shared base codebooks
    - Efficient delta updates
    """

    def __init__(self, base_path: str = "./codebooks"):
        self.base_path = Path(base_path)
        self._codebooks: Dict[str, PersistentCodebook] = {}
        self._sessions: Dict[str, LRUCodebook] = {}
        self._lock = threading.RLock()

    def get_or_create_codebook(
        self,
        name: str,
        dimensions: int = 16384
    ) -> PersistentCodebook:
        """Get or create a named codebook."""
        with self._lock:
            if name not in self._codebooks:
                path = self.base_path / name
                codebook = PersistentCodebook(str(path), dimensions)
                if path.exists():
                    codebook.load()
                self._codebooks[name] = codebook
            return self._codebooks[name]

    def create_session(
        self,
        session_id: str,
        base_codebook: str,
        max_size: int = 50000
    ) -> LRUCodebook:
        """Create session with base codebook.

        Session gets a copy of base codebook vectors with
        LRU caching for modifications.
        """
        with self._lock:
            session = LRUCodebook(max_size)

            # Copy base codebook
            if base_codebook in self._codebooks:
                base = self._codebooks[base_codebook]
                for label in base.labels:
                    vec = base.get(label)
                    if vec is not None:
                        session.put(label, vec)

            self._sessions[session_id] = session
            return session

    def get_session(self, session_id: str) -> Optional[LRUCodebook]:
        """Get session codebook."""
        return self._sessions.get(session_id)

    def close_session(self, session_id: str) -> None:
        """Close and cleanup session."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]

    def save_all(self) -> None:
        """Save all codebooks."""
        with self._lock:
            for codebook in self._codebooks.values():
                codebook.save()
