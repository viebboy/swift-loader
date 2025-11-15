"""Shared memory queue implementation for large data transfer between processes.

This module provides a queue implementation using shared memory for efficient
transfer of large data batches between parent and child processes.

* Copyright: 2023 Dat Tran
* Authors: Dat Tran
* Emails: hello@dats.bio
* Date: 2023-11-12
* Version: 0.0.1

License
-------
Apache 2.0
"""

from __future__ import annotations

from multiprocessing import shared_memory as SM
import threading
from typing import Any, Optional

from swift_loader.utils import BenchmarkProperty, Property


class SharedMemoryQueue:
    """Queue implementation using shared memory for inter-process communication.

    This queue is designed for transferring large byte arrays between processes
    efficiently. It manages free segments in shared memory and tracks sample
    indices for queue operations.

    Attributes:
        mem: Shared memory instance.
        free_segments: List of [start, stop] tuples representing free memory
            segments.
        sample_indices: List of [start, stop] tuples representing queued samples.
        is_closed: Flag indicating if the queue is closed.
        lock: Thread lock for thread-safe operations.
        logger: Logger instance for logging operations.
        benchmark_enable: Whether benchmarking is enabled.
        benchmark_logger: Logger for benchmark metrics.
        benchmark: Property object containing benchmark metrics.
        top_index: Index tracking the top element in the queue.

    Example:
        >>> queue = SharedMemoryQueue(
        ...     size=1024 * 1024,  # 1MB
        ...     size_in_bytes=1024 * 1024,
        ...     logger=logger,
        ...     benchmark=True,
        ...     benchmark_logger=benchmark_logger
        ... )
        >>> queue.put(b"data bytes")
        >>> indices = queue.get()
        >>> queue.remove(indices[0], indices[1])
    """

    def __init__(
        self,
        size: int,
        size_in_bytes: int,
        logger: Any,
        benchmark: bool,
        benchmark_logger: Any,
    ) -> None:
        """Initialize shared memory queue.

        Args:
            size: Total size of shared memory in bytes.
            size_in_bytes: Size of usable memory in bytes (typically same as size).
            logger: Logger instance for logging.
            benchmark: Whether to enable benchmarking.
            benchmark_logger: Logger for benchmark metrics.
        """
        self.mem = SM.SharedMemory(create=True, size=size)
        self.free_segments = [[0, size_in_bytes]]
        self.sample_indices = []
        # Async lock
        self.is_closed = False
        self.lock = threading.Lock()
        self.logger = logger

        # Benchmark related fields
        self.benchmark_enable = benchmark
        self.benchmark_logger = benchmark_logger

        if benchmark:
            self.benchmark = Property()
            # For ingestion measure
            self.benchmark.ingest = BenchmarkProperty(max_count=size)
            # For digest measure
            self.benchmark.digest = BenchmarkProperty(max_count=size)

        # Index to keep track of which item is on top of the queue
        # Note that calling get only increments this counter without removing
        # item. Item removal is done via remove()
        self.top_index = 0

    def name(self) -> str:
        """Return the name of the shared memory space.

        Returns:
            Name of the shared memory instance.
        """
        return self.mem.name

    def put(self, data: bytes) -> bool:
        """Try to put bytes into queue.

        Attempts to store data in the shared memory queue. Finds a free segment
        large enough to hold the data and stores it.

        Args:
            data: Byte data to store in the queue.

        Returns:
            True if successful, False if no space available.
        """
        with self.lock:
            success = False
            for idx, (start, stop) in enumerate(self.free_segments):
                if stop - start >= len(data):
                    # We have enough space
                    # Put data into memory
                    self.mem.buf[start : start + len(data)] = data
                    self.sample_indices.append((start, start + len(data)))
                    # Update free segments
                    if (stop - start) == len(data):
                        # This segment is full
                        del self.free_segments[idx]
                    else:
                        # This segment is not full
                        self.free_segments[idx][0] += len(data)
                    success = True
                    break

            return success

    def empty(self) -> bool:
        """Check if queue is empty.

        Returns:
            True if queue is empty, False otherwise.
        """
        with self.lock:
            return len(self.sample_indices) == 0

    def get(self) -> Optional[tuple[int, int]]:
        """Get info of the current top element without removing it.

        Returns information about the top element in the queue without actually
        removing the data. The data remains in shared memory until remove()
        is called.

        Returns:
            Tuple of (start, stop) indices of the byte segment in shared memory,
            or None if queue is empty or all items have been retrieved.
        """
        with self.lock:
            if len(self.sample_indices) > 0 and self.top_index < len(
                self.sample_indices
            ):
                indices = self.sample_indices[self.top_index]
                self.top_index += 1
                return indices
            return None

    def close(self) -> None:
        """Clean up the shared memory.

        Closes and unlinks the shared memory, making it unavailable for further
        use. This should be called when the queue is no longer needed.
        """
        if not self.is_closed:
            self.mem.close()
            self.mem.unlink()
            self.is_closed = True

    def remove(
        self, sample_start: Optional[int] = None, sample_stop: Optional[int] = None
    ) -> None:
        """Remove top element from queue.

        Removes a sample from the queue and updates free segments. If start and
        stop are not provided, removes the first sample in the queue.

        Args:
            sample_start: Start index of the sample to remove. If None, removes
                the first sample.
            sample_stop: Stop index of the sample to remove. If None, removes
                the first sample.

        Raises:
            ValueError: If the requested segment is not in the queue.
        """
        with self.lock:
            if len(self.sample_indices) > 0:
                if sample_start is None or sample_stop is None:
                    # If start and stop are not provided
                    sample_start, sample_stop = self.sample_indices.pop(0)
                else:
                    # If provided, need to check if exists in the queue
                    existed = False
                    idx_to_remove = None
                    for idx, (_start, _stop) in enumerate(self.sample_indices):
                        if _start == sample_start and _stop == sample_stop:
                            existed = True
                            idx_to_remove = idx
                            break

                    if not existed:
                        # If not existed
                        error_msg = (
                            f"Requested segment with start={sample_start} and "
                            f"stop={sample_stop} is not in the queue"
                        )
                        self.logger.error(
                            error_msg,
                            source="mem_queue.SharedMemoryQueue.remove()",
                        )
                        raise ValueError(error_msg)

                    self.sample_indices.pop(idx_to_remove)

                self.top_index -= 1
                # Need to update free segments
                if len(self.free_segments) == 0:
                    # If there's no free segment
                    self.free_segments.append([sample_start, sample_stop])
                else:
                    # There is at least 1 free segment
                    merged = False
                    for idx, (segment_start, segment_stop) in enumerate(
                        self.free_segments
                    ):
                        if sample_start == segment_stop:
                            # Merge with the previous segment
                            self.free_segments[idx][1] = sample_stop
                            merged = True
                            break
                        elif sample_stop == segment_start:
                            # Merge with the next segment
                            self.free_segments[idx][0] = sample_start
                            merged = True
                            break

                    if not merged:
                        # If not merged with any segment
                        self.free_segments.append([sample_start, sample_stop])
                    else:
                        # Merge any consecutive segments
                        self.free_segments = self.merge_segments(self.free_segments)

    def merge_segments(self, segments: list[list[int]]) -> list[list[int]]:
        """Merge consecutive free segments.

        Args:
            segments: List of [start, stop] segment tuples.

        Returns:
            List of merged segments with consecutive segments combined.
        """
        # Sort by start
        segments.sort(key=lambda x: x[0])

        merged_segments = []
        current_segment = segments[0]

        for segment in segments[1:]:
            if current_segment[1] == segment[0]:
                current_segment = [current_segment[0], segment[1]]
            else:
                merged_segments.append(current_segment)
                current_segment = segment

        merged_segments.append(current_segment)

        return merged_segments

    def __len__(self) -> int:
        """Get the number of items in the queue.

        Returns:
            Number of items currently in the queue.
        """
        with self.lock:
            return len(self.sample_indices)
