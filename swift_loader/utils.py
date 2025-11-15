"""Utility functions and classes for SwiftLoader.

This module provides utility functions for collation, device transfer, temporary
file management, and benchmarking.

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

import os
import random
import string
import tempfile
import time
from typing import Any, Callable

import numpy as np


class Property:
    """Namespace to hold properties.

    This is a simple container class used to create namespaces for storing
    related properties together.

    Example:
        >>> prop = Property()
        >>> prop.value = 42
        >>> prop.name = "test"
    """

    def __init__(self) -> None:
        """Initialize an empty property namespace."""
        pass


class BenchmarkProperty:
    """Abstraction to hold benchmark properties.

    Intended to be used with TimeMeasure context manager for measuring
    performance metrics.

    Attributes:
        time: Accumulated time in seconds.
        count: Number of measurements taken.
        max_count: Maximum number of measurements before reset.

    Example:
        >>> bench = BenchmarkProperty(max_count=100)
        >>> with TimeMeasure(bench, logger, "task"):
        ...     # Do work
        ...     pass
    """

    def __init__(self, max_count: int) -> None:
        """Initialize benchmark property.

        Args:
            max_count: Maximum number of measurements before automatic reset.
        """
        self.time = 0.0
        self.count = 0
        self.max_count = max_count

    def reset(self) -> None:
        """Reset benchmark counters."""
        self.time = 0.0
        self.count = 0


class TimeMeasure:
    """Context manager to measure time taken for a given block of code.

    Benchmark results are logged via logger when max_count is reached.

    Attributes:
        counter: BenchmarkProperty instance to track measurements.
        logger: Logger instance for output.
        task_name: Name of the task being measured.

    Example:
        >>> bench = BenchmarkProperty(max_count=100)
        >>> with TimeMeasure(bench, logger, "data_loading"):
        ...     load_data()
    """

    def __init__(self, counter: BenchmarkProperty, logger: Any, task_name: str) -> None:
        """Initialize time measure context manager.

        Args:
            counter: BenchmarkProperty to track measurements.
            logger: Logger instance for output.
            task_name: Name of the task being measured.
        """
        self.counter = counter
        self.logger = logger
        self.task_name = task_name
        self.start = None

    def __enter__(self) -> TimeMeasure:
        """Enter context manager and start timing.

        Returns:
            Self instance.
        """
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context manager and record timing.

        Args:
            *args: Exception information (if any).
        """
        self.counter.count += 1
        self.counter.time += time.perf_counter() - self.start
        if self.counter.count == self.counter.max_count:
            fps = self.counter.count / self.counter.time
            self.logger.info(f"FPS={fps}", benchmark_task=self.task_name)
            self.counter.reset()


class ThroughputMeasure:
    """Convenient class to measure throughput of a task.

    Measures throughput (items per second) for a task by tracking the number
    of items processed over time.

    Attributes:
        logger: Logger instance for output.
        task_name: Name of the task being measured.
        max_count: Number of items to process before logging throughput.
        cur_count: Current count of items processed.
        start_time: Start time of measurement period.
        stop_time: Stop time of measurement period.

    Example:
        >>> measure = ThroughputMeasure(max_count=100, logger=logger, task_name="processing")
        >>> for item in items:
        ...     measure.start()
        ...     process(item)
        ...     measure.count()
    """

    def __init__(self, max_count: int, logger: Any, task_name: str) -> None:
        """Initialize throughput measure.

        Args:
            max_count: Number of items to process before logging.
            logger: Logger instance for output.
            task_name: Name of the task being measured.
        """
        self.logger = logger
        self.task_name = task_name
        self.max_count = max_count
        self.cur_count = 0
        self.start_time = None
        self.stop_time = None

    def start(self) -> None:
        """Start timing. Should be called at the start of the action."""
        if self.cur_count == 0:
            self.start_time = time.perf_counter()

    def count(self) -> None:
        """Increment the counter and log throughput if max_count reached."""
        self.cur_count += 1
        if self.cur_count == self.max_count:
            self.stop_time = time.perf_counter()
            throughput = self.max_count / (self.stop_time - self.start_time)
            self.logger.debug(
                f"Throughput: {throughput} FPS",
                benchmark_task=self.task_name,
            )
            self.cur_count = 0


class DummyTimeMeasure:
    """Dummy class to act as TimeMeasure when benchmarking is disabled.

    Provides the same interface as TimeMeasure but does nothing, allowing
    code to work with or without benchmarking enabled.

    Example:
        >>> if benchmark:
        ...     measure = TimeMeasure(counter, logger, "task")
        ... else:
        ...     measure = DummyTimeMeasure(counter, logger, "task")
        >>> with measure:
        ...     do_work()
    """

    def __init__(self, counter: Any, logger: Any, task_name: str) -> None:
        """Initialize dummy time measure.

        Args:
            counter: Unused (for interface compatibility).
            logger: Unused (for interface compatibility).
            task_name: Unused (for interface compatibility).
        """
        pass

    def __enter__(self) -> DummyTimeMeasure:
        """Enter context manager (no-op).

        Returns:
            Self instance.
        """
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context manager (no-op).

        Args:
            *args: Exception information (if any).
        """
        pass


def get_default_collate_fn() -> Callable:
    """Return default collate function.

    The returned collate function can work with nested lists of numpy arrays
    or torch tensors. It concatenates samples along the first dimension.

    Returns:
        Collate function that takes a list of samples and returns a batch.

    Example:
        >>> collate_fn = get_default_collate_fn()
        >>> batch = collate_fn([sample1, sample2, sample3])
    """
    import torch

    def concatenate_list(inputs: list) -> Any:
        """Concatenate a list of nested lists of numpy arrays or torch tensors.

        This function is used to concatenate a list of nested lists of numpy
        arrays (or torch tensors) to a nested list of numpy arrays or torch
        tensors.

        For example:
            inputs = [x_1, x_2, ..., x_N]
            each x_i is a nested list of numpy arrays
            for example, x_i = [np.ndarray, [np.ndarray, np.ndarray]]

            the outputs should be:
            outputs = [np.ndarray, [np.ndarray, np.ndarray]]
            in which each np.ndarray is a concatenation of the corresponding
            element from the same position.

        Args:
            inputs: List of samples, where each sample can be a nested structure
                of numpy arrays or torch tensors.

        Returns:
            Batch with same structure as input samples, but with concatenated
            arrays/tensors.

        Raises:
            RuntimeError: If concatenation fails.
        """

        def _create_nested_list(x: Any, data: list) -> None:
            """Create nested list structure based on input structure.

            Args:
                x: Sample to analyze structure.
                data: List to populate with nested structure.
            """
            if isinstance(x, (list, tuple)):
                # There is a nested structure
                # Create sub list for each item
                for _ in range(len(x)):
                    data.append([])
                for idx, item in enumerate(x):
                    _create_nested_list(item, data[idx])
            else:
                # Do nothing to the input list
                pass

        def _process_sample(x: Any, data: list) -> None:
            """Process a sample and add to data structure.

            Args:
                x: Sample to process.
                data: Data structure to populate.
            """
            if isinstance(x, (list, tuple)):
                for idx, item in enumerate(x):
                    _process_sample(item, data[idx])
            else:
                data.append(x)

        # Create an empty nested list based on the structure of the 1st sample
        outputs = []
        _create_nested_list(inputs[0], outputs)

        # Then process each sample
        for sample in inputs:
            _process_sample(sample, outputs)

        def _concatenate_item(x: list) -> Any:
            """Concatenate items in a list.

            Args:
                x: List of items to concatenate.

            Returns:
                Concatenated result.

            Raises:
                RuntimeError: If concatenation fails.
            """
            if isinstance(x, list) and isinstance(x[0], torch.Tensor):
                if torch.numel(x[0]) == 1:
                    result = torch.tensor(x)
                else:
                    result = torch.cat([item.unsqueeze(0) for item in x], dim=0)
                return result
            elif isinstance(x, list) and isinstance(x[0], np.ndarray):
                return np.concatenate([np.expand_dims(item, 0) for item in x], axis=0)
            elif isinstance(x, list) and isinstance(x[0], (float, int)):
                return np.asarray(x).flatten()
            elif isinstance(x, list) and isinstance(x[0], list):
                return [_concatenate_item(item) for item in x]
            else:
                raise RuntimeError(
                    "Failed to concatenate a list of samples generated from dataset"
                )

        return _concatenate_item(outputs)

    return concatenate_list


def get_temp_file(length: int = 32) -> str:
    """Get temporary file name with random name.

    Args:
        length: Length of random filename. Must be between 1 and 255.
            Defaults to 32.

    Returns:
        Full path to temporary file (file is not created, only path returned).

    Raises:
        AssertionError: If length is out of valid range.
    """
    assert 0 < length < 256, "length must be between 1 and 255"
    alphabet = list(string.ascii_lowercase)
    random_name = [random.choice(alphabet) for _ in range(length)]
    random_name = os.path.join(tempfile.gettempdir(), "".join(random_name))
    return random_name


def get_default_to_device(half: bool = False) -> Callable:
    """Get default function that moves data to a device.

    Args:
        half: Whether to convert to half precision (float16). Defaults to False.

    Returns:
        Function that takes (input_data, device) and returns data on device.

    Example:
        >>> to_device = get_default_to_device(half=False)
        >>> data_on_gpu = to_device(data, "cuda:0")
    """
    import torch

    def to_device(input_data: Any, device: Any) -> Any:
        """Move input data to specified device.

        Recursively handles nested structures (lists, tuples) and converts
        numpy arrays to torch tensors if needed.

        Args:
            input_data: Data to move to device. Can be torch.Tensor, np.ndarray,
                or nested structure of these.
            device: Target device (e.g., "cuda:0" or torch.device).

        Returns:
            Data moved to device, with same structure as input.
        """
        if isinstance(input_data, (list, tuple)):
            # There is a nested structure
            # Create sub list for each item
            output_data = []
            for item in input_data:
                output_data.append(to_device(item, device))
            return output_data
        elif isinstance(input_data, np.ndarray):
            if half:
                return torch.as_tensor(input_data).half().to(device, non_blocking=True)
            else:
                return torch.as_tensor(input_data).to(device, non_blocking=True)
        else:
            try:
                if half:
                    return input_data.half().to(device, non_blocking=True)
                else:
                    return input_data.to(device, non_blocking=True)
            except Exception:
                return input_data

    return to_device


def shuffle_indices(
    seed: int, start_idx: int, stop_idx: int, nearby_shuffle: int
) -> list[int]:
    """Shuffle indices with optional nearby shuffling.

    This function shuffles indices and allows nearby shuffling, which shuffles
    only nearby indices. Nearby shuffling is useful if we want to reduce
    overheads in seeking binary files while still maintaining some randomness.

    Args:
        seed: Random seed for shuffling.
        start_idx: Start index (inclusive).
        stop_idx: Stop index (exclusive).
        nearby_shuffle: Window size for nearby shuffling. If 0, performs full
            shuffle. If > 0, only shuffles within windows of this size.

    Returns:
        List of shuffled indices.
    """
    random.seed(seed)
    indices = list(range(start_idx, stop_idx))
    if nearby_shuffle > 0:
        i = random.choice(indices[1:-1])
        indices_ = indices[i:] + indices[:i]
        indices = []
        start_index = 0
        N = len(indices_)
        while len(indices) < N:
            stop_index = min(N, start_index + nearby_shuffle)
            sub_indices = indices_[start_index:stop_index]
            random.shuffle(sub_indices)
            indices.extend(sub_indices)
            start_index = stop_index
    else:
        random.shuffle(indices)

    return indices
