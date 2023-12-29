"""
utils.py: package utilities
---------------------------


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
from typing import Any, Callable
import time
import random
import tempfile
import numpy as np
import string
import os


class Property:
    """
    Namespace to hold properties
    """

    def __init__(self):
        pass


class BenchmarkProperty:
    """
    Abstraction to hold benchmark properties
    Intended to be used wtih TimeMeasure context
    """

    def __init__(self, max_count: int):
        self.time = 0
        self.count = 0
        self.max_count = max_count

    def reset(self):
        self.time = 0
        self.count = 0


class TimeMeasure:
    """
    Context manager to measure time taken for a given block of code
    Benchmark results are logged via logger
    """

    def __init__(self, counter: BenchmarkProperty, logger: Any, task_name: str):
        self.counter = counter
        self.logger = logger
        self.task_name = task_name

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, *args):
        self.counter.count += 1
        self.counter.time += time.perf_counter() - self.start
        if self.counter.count == self.counter.max_count:
            fps = self.counter.count / self.counter.time
            self.logger.info(f"FPS={fps}", benchmark_task=self.task_name)
            self.counter.reset()


class ThroughputMeasure:
    """
    Convenient class to measure throughput of a task
    """

    def __init__(self, max_count: int, logger: Any, task_name: str):
        self.logger = logger
        self.task_name = task_name
        self.max_count = max_count
        self.cur_count = 0

    def start(self):
        """
        This should be called at the start of the action
        """
        if self.cur_count == 0:
            self.start_time = time.perf_counter()

    def count(self):
        """
        Increment the counter
        """
        self.cur_count += 1
        if self.cur_count == self.max_count:
            self.stop_time = time.perf_counter()
            self.logger.debug(
                f"Throughput: {self.max_count/(self.stop_time - self.start_time)} FPS",
                benchmark_task=self.task_name,
            )
            self.cur_count = 0


class DummyTimeMeasure:
    """
    Dummy class to act as TimeMeasure
    """

    def __init__(self, counter, logger, task_name):
        pass

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


def get_default_collate_fn() -> Callable:
    """
    Return defaullt collate_fn
    The returned collate_fn can work with nested list of numpy arrays or torch tensors
    """
    import torch

    def concatenate_list(inputs):
        """
        This function is used to concatenate
        a list of nested lists of numpy array (or torch) to a nested list of numpy arrays or torch

        For example,
        inputs = [x_1, x_2, ..., x_N]
        each x_i is a nested list of numpy arrays
        for example,
        x_i = [np.ndarray, [np.ndarray, np.ndarray]]

        the outputs should be
        outputs = [np.ndarray, [np.ndarray, np.ndarray]]
        in which each np.ndarray is a concatenation of the corresponding element
        from the same position
        """

        def _create_nested_list(x, data):
            if isinstance(x, (list, tuple)):
                # there is a nested structure
                # create sub list for each item
                for _ in range(len(x)):
                    data.append([])
                for idx, item in enumerate(x):
                    _create_nested_list(item, data[idx])
            else:
                # do nothing to the input list
                next

        def _process_sample(x, data):
            if isinstance(x, (list, tuple)):
                for idx, item in enumerate(x):
                    _process_sample(item, data[idx])
            else:
                data.append(x)

        # create an empty nested list based on the structure of the 1st sample
        outputs = []
        _create_nested_list(inputs[0], outputs)

        # then process each sample
        for sample in inputs:
            _process_sample(sample, outputs)

        def _concatenate_item(x):
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


def get_temp_file(length=32) -> str:
    """
    Get temporary file name given name length
    """
    assert 0 < length < 256
    alphabet = list(string.ascii_lowercase)
    random_name = [random.choice(alphabet) for _ in range(length)]
    random_name = os.path.join(tempfile.gettempdir(), "".join(random_name))
    return random_name


def get_default_to_device(half=False) -> Callable:
    """
    Get default function that moves data to a device
    """
    import torch

    def to_device(input_data, device):
        if isinstance(input_data, (list, tuple)):
            # there is a nested structure
            # create sub list for each item
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


def shuffle_indices(seed, start_idx, stop_idx, nearby_shuffle) -> list[int]:
    """
    Function to shuffle indices
    This also allows nearby shuffling, which shuffles only nearby indices
    Nearby shuffling is useful if we want to reduce overheads in seeking binary files
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
