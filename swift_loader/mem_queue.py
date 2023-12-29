"""
mem_queue.py: queue implemented on shared_memory
this is intended for large data telcom between processes
--------------------------------------------------------


* Copyright: 2023 Dat Tran
* Authors: Dat Tran
* Emails: hello@dats.bio
* Date: 2023-11-12
* Version: 0.0.1


This is part of the swift_loader package


License
-------
Apache 2.0

"""


from __future__ import annotations
from typing import Any
from multiprocessing import shared_memory as SM
import threading
from swift_loader.utils import BenchmarkProperty, Property


class SharedMemoryQueue:
    """
    Queue implementation on shared memory
    """

    def __init__(
        self,
        size: int,
        size_in_bytes: int,
        logger: Any,
        benchmark: bool,
        benchmark_logger: Any,
    ):
        self.mem = SM.SharedMemory(create=True, size=size)
        self.free_segments = [[0, size_in_bytes]]
        self.sample_indices = []
        # async lock
        self.is_closed = False
        self.lock = threading.Lock()
        self.logger = logger

        # benchmark related fields
        self.benchmark_enable = benchmark
        self.benchmark_logger = benchmark_logger

        if benchmark:
            self.benchmark = Property()
            # for ingestion measure
            self.benchmark.ingest = BenchmarkProperty(max_count=size)
            # for digest measure
            self.benchmark.digest = BenchmarkProperty(max_count=size)

        # index to keep track of which item is on top of the queue
        # note that calling get only increment this counter without removing
        # item. Item removable is done via remove()
        self.top_index = 0

    def name(self) -> str:
        """
        Return the name of the shared memory space
        """
        return self.mem.name

    def put(self, data: bytes) -> bool:
        """
        Try to put bytes into queue
        Return True if successful
        """

        with self.lock:
            success = False
            for idx, (start, stop) in enumerate(self.free_segments):
                if stop - start >= len(data):
                    # we have enough space
                    # put data into memory
                    self.mem.buf[start : start + len(data)] = data
                    self.sample_indices.append((start, start + len(data)))
                    # update free segments
                    if (stop - start) == len(data):
                        # this segment is full
                        del self.free_segments[idx]
                    else:
                        # this segment is not full
                        self.free_segments[idx][0] += len(data)
                    success = True
                    break

            return success

    def empty(self) -> bool:
        """
        Check if queue is empty
        """
        with self.lock:
            return len(self.sample_indices) == 0

    def get(self) -> tuple[int, int]:
        """
        Return info of the current top element in the queue without removing the content
        Info is a 2-tuple containing the start and stop indices of the byte segment in shared memory
        """
        with self.lock:
            if len(self.sample_indices) > 0 and self.top_index < len(
                self.sample_indices
            ):
                indices = self.sample_indices[self.top_index]
                self.top_index += 1
                return indices

    def close(self):
        """
        Clean up the shared memory
        """
        if not self.is_closed:
            self.mem.close()
            self.mem.unlink()
            self.is_closed = True

    def remove(self, sample_start=None, sample_stop=None):
        """
        Remove top element in queue
        """
        with self.lock:
            if len(self.sample_indices) > 0:
                if sample_start is None or sample_stop is None:
                    # if start and stop are not provided
                    sample_start, sample_stop = self.sample_indices.pop(0)
                else:
                    # if provided, need to check if exists in the queue
                    existed = False
                    idx_to_remove = None
                    for idx, (_start, _stop) in enumerate(self.sample_indices):
                        if _start == sample_start and _stop == sample_stop:
                            existed = True
                            idx_to_remove = idx
                            break

                    if not existed:
                        # if not existed
                        self.logger.error(
                            (
                                f"Requested segment with start={sample_start} and stop={sample_stop} "
                                "is not in the queue"
                            ),
                            source="mem_queue.SharedMemoryQueue.remove()",
                        )
                        raise ValueError(
                            (
                                f"Requested segment with start={sample_start} and stop={sample_stop} "
                                "is not in the queue"
                            )
                        )

                    self.sample_indices.pop(idx_to_remove)

                self.top_index -= 1
                # need to update free segments
                if len(self.free_segments) == 0:
                    # if there's no free segment
                    self.free_segments.append([sample_start, sample_stop])
                else:
                    # there is at least 1 free segment
                    merged = False
                    for idx, (segment_start, segment_stop) in enumerate(
                        self.free_segments
                    ):
                        if sample_start == segment_stop:
                            # merge with the previous segment
                            self.free_segments[idx][1] = sample_stop
                            merged = True
                            break
                        elif sample_stop == segment_start:
                            # merge with the next segment
                            self.free_segments[idx][0] = sample_start
                            merged = True
                            break

                    if not merged:
                        # if not merged with any segment
                        self.free_segments.append([sample_start, sample_stop])
                    else:
                        # merge any consecutive segments
                        self.free_segments = self.merge_segments(self.free_segments)

    def merge_segments(self, segments):
        """
        Merge existing free segments
        """
        # sort start
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

    def __len__(self):
        with self.lock:
            return len(self.sample_indices)
