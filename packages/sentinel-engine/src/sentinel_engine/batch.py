"""
engine/batch.py - Batch and Stream Processing Utilities

Provides efficient batch processing for large datasets:
- Chunked processing to manage memory
- Parallel batch execution
- Streaming for continuous data
- Progress tracking
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Iterator, Callable, TypeVar, Generic
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import time
import logging
from queue import Queue
import threading

import torch

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class BatchResult(Generic[T]):
    """Result from batch processing."""
    batch_idx: int
    results: List[T]
    elapsed_ms: float
    errors: List[str] = field(default_factory=list)


@dataclass
class StreamStats:
    """Statistics for stream processing."""
    processed: int = 0
    errors: int = 0
    total_elapsed_ms: float = 0.0
    avg_latency_ms: float = 0.0
    throughput_per_sec: float = 0.0


class BatchProcessor(Generic[T, R]):
    """Parallel batch processor with chunking.

    Processes large datasets in chunks with parallel execution.

    Example:
        processor = BatchProcessor(
            process_fn=analyze_entity,
            batch_size=1000,
            max_workers=4
        )

        results = processor.process(entities)
    """

    def __init__(
        self,
        process_fn: Callable[[T], R],
        batch_size: int = 1000,
        max_workers: int = 4,
        use_processes: bool = False
    ):
        """Initialize batch processor.

        Args:
            process_fn: Function to apply to each item
            batch_size: Items per batch
            max_workers: Parallel workers
            use_processes: Use processes instead of threads
        """
        self.process_fn = process_fn
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.use_processes = use_processes

        self._stats = {
            "batches_processed": 0,
            "items_processed": 0,
            "errors": 0,
            "total_time_ms": 0,
        }

    def _chunk(self, items: List[T]) -> Iterator[List[T]]:
        """Split items into chunks."""
        for i in range(0, len(items), self.batch_size):
            yield items[i:i + self.batch_size]

    def _process_batch(self, batch_idx: int, batch: List[T]) -> BatchResult[R]:
        """Process a single batch."""
        start = time.time()
        results = []
        errors = []

        for item in batch:
            try:
                result = self.process_fn(item)
                results.append(result)
            except Exception as e:
                errors.append(str(e))
                logger.warning(f"Error processing item: {e}")

        elapsed = (time.time() - start) * 1000

        return BatchResult(
            batch_idx=batch_idx,
            results=results,
            elapsed_ms=elapsed,
            errors=errors
        )

    def process(
        self,
        items: List[T],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[R]:
        """Process all items in parallel batches.

        Args:
            items: Items to process
            progress_callback: Optional callback(processed, total)

        Returns:
            All results in order
        """
        total_start = time.time()
        batches = list(self._chunk(items))
        all_results: Dict[int, List[R]] = {}
        total_errors = []

        ExecutorClass = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor

        with ExecutorClass(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._process_batch, i, batch): i
                for i, batch in enumerate(batches)
            }

            completed = 0
            for future in as_completed(futures):
                batch_idx = futures[future]
                try:
                    result = future.result()
                    all_results[batch_idx] = result.results
                    total_errors.extend(result.errors)
                except Exception as e:
                    logger.error(f"Batch {batch_idx} failed: {e}")
                    total_errors.append(str(e))

                completed += 1
                if progress_callback:
                    progress_callback(completed * self.batch_size, len(items))

        # Reconstruct ordered results
        ordered_results = []
        for i in range(len(batches)):
            ordered_results.extend(all_results.get(i, []))

        # Update stats
        self._stats["batches_processed"] += len(batches)
        self._stats["items_processed"] += len(items)
        self._stats["errors"] += len(total_errors)
        self._stats["total_time_ms"] += (time.time() - total_start) * 1000

        return ordered_results

    @property
    def stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return dict(self._stats)


class StreamProcessor(Generic[T, R]):
    """Continuous stream processor.

    Processes items from a queue with configurable throughput.

    Example:
        processor = StreamProcessor(
            process_fn=analyze_entity,
            max_queue_size=10000
        )

        # Start processing
        processor.start()

        # Add items
        for entity in entities:
            processor.put(entity)

        # Get results
        for result in processor.results():
            print(result)

        # Stop
        processor.stop()
    """

    def __init__(
        self,
        process_fn: Callable[[T], R],
        max_queue_size: int = 10000,
        num_workers: int = 4,
        result_buffer_size: int = 1000
    ):
        """Initialize stream processor.

        Args:
            process_fn: Function to apply to each item
            max_queue_size: Maximum input queue size
            num_workers: Number of worker threads
            result_buffer_size: Result buffer size
        """
        self.process_fn = process_fn
        self.max_queue_size = max_queue_size
        self.num_workers = num_workers

        self._input_queue: Queue[Optional[T]] = Queue(max_queue_size)
        self._result_queue: Queue[R] = Queue(result_buffer_size)
        self._workers: List[threading.Thread] = []
        self._running = False
        self._stats = StreamStats()
        self._stats_lock = threading.Lock()

    def _worker(self) -> None:
        """Worker thread function."""
        while self._running:
            try:
                item = self._input_queue.get(timeout=0.1)
                if item is None:
                    break

                start = time.time()
                try:
                    result = self.process_fn(item)
                    self._result_queue.put(result)

                    elapsed = (time.time() - start) * 1000
                    with self._stats_lock:
                        self._stats.processed += 1
                        self._stats.total_elapsed_ms += elapsed
                        self._stats.avg_latency_ms = (
                            self._stats.total_elapsed_ms / self._stats.processed
                        )

                except Exception as e:
                    logger.warning(f"Stream processing error: {e}")
                    with self._stats_lock:
                        self._stats.errors += 1

            except Exception:
                continue

    def start(self) -> None:
        """Start stream processing."""
        if self._running:
            return

        self._running = True
        self._workers = []

        for i in range(self.num_workers):
            worker = threading.Thread(target=self._worker, daemon=True)
            worker.start()
            self._workers.append(worker)

        logger.info(f"Started stream processor with {self.num_workers} workers")

    def stop(self, wait: bool = True) -> None:
        """Stop stream processing.

        Args:
            wait: Wait for workers to finish
        """
        self._running = False

        # Send stop signals
        for _ in self._workers:
            self._input_queue.put(None)

        if wait:
            for worker in self._workers:
                worker.join(timeout=5.0)

        self._workers = []
        logger.info("Stopped stream processor")

    def put(self, item: T, block: bool = True, timeout: Optional[float] = None) -> bool:
        """Add item to processing queue.

        Args:
            item: Item to process
            block: Block if queue full
            timeout: Timeout for blocking

        Returns:
            True if item was added
        """
        try:
            self._input_queue.put(item, block=block, timeout=timeout)
            return True
        except Exception:
            return False

    def results(self, timeout: float = 0.1) -> Iterator[R]:
        """Iterate over available results.

        Args:
            timeout: Timeout for each get

        Yields:
            Processed results
        """
        while True:
            try:
                result = self._result_queue.get(timeout=timeout)
                yield result
            except Exception:
                break

    def get_result(self, timeout: float = 1.0) -> Optional[R]:
        """Get single result.

        Args:
            timeout: Timeout for get

        Returns:
            Result or None if timeout
        """
        try:
            return self._result_queue.get(timeout=timeout)
        except Exception:
            return None

    @property
    def stats(self) -> StreamStats:
        """Get current statistics."""
        with self._stats_lock:
            stats = StreamStats(
                processed=self._stats.processed,
                errors=self._stats.errors,
                total_elapsed_ms=self._stats.total_elapsed_ms,
                avg_latency_ms=self._stats.avg_latency_ms,
            )

            if stats.total_elapsed_ms > 0:
                stats.throughput_per_sec = (
                    stats.processed / (stats.total_elapsed_ms / 1000)
                )

            return stats

    @property
    def queue_size(self) -> int:
        """Current input queue size."""
        return self._input_queue.qsize()

    @property
    def result_count(self) -> int:
        """Number of results available."""
        return self._result_queue.qsize()
