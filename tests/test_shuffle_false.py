"""
Test cases for shuffle=False configurations

These tests ensure:
1. Data is returned in the correct order (no shuffling)
2. All samples are covered
3. No deadlocks occur
"""

import numpy as np
import pytest
from swift_loader import SwiftLoader
import math

# Check if numpy is available
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


class IndexDataset:
    """Dataset that returns the index as data for easy order verification"""

    def __init__(self, size: int):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int) -> int:
        if idx >= self.size or idx < 0:
            raise IndexError(
                f"Index {idx} out of range for dataset of size {self.size}"
            )
        return idx


def verify_consumer_order(
    consumer_samples_list: list, dataset_size: int, nb_consumer: int
):
    """
    Verify that samples from consumers come in order (consumer 0, then 1, etc.)

    When shuffle=False with multiple consumers:
    - Consumer 0: samples 0 to ceil(dataset_size/nb_consumer) - 1
    - Consumer 1: samples ceil(dataset_size/nb_consumer) to 2*ceil(...) - 1
    - etc.

    When we iterate through consumers in order, the combined samples should be
    in sequential order [0, 1, 2, ..., dataset_size-1].
    """
    consumer_size = int(math.ceil(dataset_size / nb_consumer))
    expected_samples = list(range(dataset_size))

    # Verify each consumer has the correct samples
    for consumer_idx, consumer_samples in enumerate(consumer_samples_list):
        consumer_start = consumer_idx * consumer_size
        consumer_stop = min(consumer_start + consumer_size, dataset_size)
        expected_consumer_samples = list(range(consumer_start, consumer_stop))

        if set(consumer_samples) != set(expected_consumer_samples):
            missing = set(expected_consumer_samples) - set(consumer_samples)
            extra = set(consumer_samples) - set(expected_consumer_samples)
            raise AssertionError(
                f"Consumer {consumer_idx} sample mismatch. "
                f"Expected: {expected_consumer_samples}, Got: {consumer_samples}. "
                f"Missing: {missing}, Extra: {extra}"
            )

        # Verify order within consumer
        if consumer_samples != expected_consumer_samples:
            first_diff = next(
                (
                    i
                    for i, (a, b) in enumerate(
                        zip(consumer_samples, expected_consumer_samples)
                    )
                    if a != b
                ),
                None,
            )
            raise AssertionError(
                f"Consumer {consumer_idx} order incorrect. "
                f"First difference at index {first_diff}: "
                f"expected {expected_consumer_samples[first_diff]}, "
                f"got {consumer_samples[first_diff]}"
            )

    # Verify combined samples are in order
    combined_samples = []
    for consumer_samples in consumer_samples_list:
        combined_samples.extend(consumer_samples)

    if combined_samples != expected_samples:
        first_diff = next(
            (
                i
                for i, (a, b) in enumerate(zip(combined_samples, expected_samples))
                if a != b
            ),
            None,
        )
        raise AssertionError(
            f"Combined consumer order incorrect. "
            f"First difference at index {first_diff}: "
            f"expected {expected_samples[first_diff]}, "
            f"got {combined_samples[first_diff]}"
        )


def verify_order(samples: list, dataset_size: int, batch_size: int, nb_workers: int):
    """
    Verify that samples are in correct order when shuffle=False

    When shuffle=False with multiple workers, batches are interleaved:
    - Worker 0: batches 0, 3, 6, ... (samples: 0-1, 6-7, 12-13, ...)
    - Worker 1: batches 1, 4, 7, ... (samples: 2-3, 8-9, 14-15, ...)
    - Worker 2: batches 2, 5, 8, ... (samples: 4-5, 10-11, 16-17, ...)

    With OrderDataQueue, batches should come in order: 0, 1, 2, 3, 4, ...
    So samples should be: [0,1], [2,3], [4,5], [6,7], [8,9], ...
    """
    expected_samples = list(range(dataset_size))

    # Check all samples are present
    if len(samples) != len(expected_samples):
        raise AssertionError(
            f"Expected {len(expected_samples)} samples, got {len(samples)}"
        )

    if set(samples) != set(expected_samples):
        missing = set(expected_samples) - set(samples)
        extra = set(samples) - set(expected_samples)
        raise AssertionError(f"Sample mismatch. Missing: {missing}, Extra: {extra}")

    # For single worker, order should be strictly sequential
    if nb_workers == 1:
        if samples != expected_samples:
            first_diff = next(
                (
                    i
                    for i, (a, b) in enumerate(zip(samples, expected_samples))
                    if a != b
                ),
                None,
            )
            raise AssertionError(
                f"Order incorrect for single worker. "
                f"First difference at index {first_diff}: "
                f"expected {expected_samples[first_diff]}, got {samples[first_diff]}"
            )
        return

    # For multiple workers with OrderDataQueue, batches should be interleaved
    # in the correct order: batch 0, batch 1, batch 2, batch 3, ...
    # This means samples should appear as: [0,1], [2,3], [4,5], [6,7], ...

    num_batches = (dataset_size + batch_size - 1) // batch_size

    # Verify batches are in order
    sample_pos = 0
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, dataset_size)
        expected_batch = list(range(batch_start, batch_end))

        # Get the actual batch from samples
        if sample_pos + len(expected_batch) > len(samples):
            raise AssertionError(
                f"Not enough samples for batch {batch_idx}. "
                f"Expected {len(expected_batch)} samples starting at position {sample_pos}"
            )

        actual_batch = samples[sample_pos : sample_pos + len(expected_batch)]

        # Verify the batch matches
        if actual_batch != expected_batch:
            raise AssertionError(
                f"Batch {batch_idx} order incorrect. "
                f"Expected: {expected_batch}, Got: {actual_batch}"
            )

        sample_pos += len(expected_batch)


def extract_samples_from_batch(batch):
    """Extract sample indices from a batch"""
    if isinstance(batch, np.ndarray):
        if batch.ndim == 1:
            batch_list = batch.tolist()
        else:
            # Multi-dimensional array - from RandomNumpyDataset
            batch_size_actual = batch.shape[0]
            # For IndexDataset, we can't extract meaningful indices from random data
            # So we'll just count the number of samples
            batch_list = list(range(batch_size_actual))
    elif hasattr(batch, "tolist"):
        batch_list = batch.tolist()
    elif isinstance(batch, (list, tuple)):
        batch_list = list(batch)
    else:
        try:
            batch_list = [int(x) for x in batch]
        except (ValueError, TypeError):
            batch_list = [batch]

    # Flatten if nested (but only if elements are lists/tuples of numbers, not numpy arrays)
    if (
        batch_list
        and isinstance(batch_list[0], (list, tuple))
        and not isinstance(batch_list[0], np.ndarray)
        and not isinstance(batch_list[0], (int, float))
    ):
        try:
            batch_list = [
                item
                for sublist in batch_list
                for item in sublist
            ]
        except (TypeError, ValueError):
            pass

    return batch_list


class TestShuffleFalse:
    """Test cases for shuffle=False configurations"""

    @pytest.mark.parametrize("multithread", [False, True])
    def test_single_consumer_single_worker(self, multithread):
        """Test: 1 consumer, 1 worker, shuffle=False"""
        dataset_size = 47  # Prime number (uneven)
        batch_size = 5
        num_epochs = 5

        loader = SwiftLoader(
            dataset_class=IndexDataset,
            dataset_kwargs={"size": dataset_size},
            batch_size=batch_size,
            nb_consumer=1,
            worker_per_consumer=1,
            shuffle=False,
            multithread=multithread,
            data_queue_size=10,
        )
        loader.start(consumer_index=0)

        try:
            for epoch in range(num_epochs):
                epoch_samples = []

                for batch in loader:
                    batch_list = extract_samples_from_batch(batch)
                    epoch_samples.extend(batch_list)

                verify_order(epoch_samples, dataset_size, batch_size, 1)

        finally:
            loader.close()

    @pytest.mark.parametrize("multithread", [False, True])
    def test_single_consumer_two_workers(self, multithread):
        """Test: 1 consumer, 2 workers, shuffle=False"""
        dataset_size = 43  # Prime: 43/2 = 21.5 -> workers get 22, 21
        batch_size = 5
        num_epochs = 5

        loader = SwiftLoader(
            dataset_class=IndexDataset,
            dataset_kwargs={"size": dataset_size},
            batch_size=batch_size,
            nb_consumer=1,
            worker_per_consumer=2,
            shuffle=False,
            multithread=multithread,
            data_queue_size=10,
        )
        loader.start(consumer_index=0)

        try:
            for epoch in range(num_epochs):
                epoch_samples = []

                for batch in loader:
                    batch_list = extract_samples_from_batch(batch)
                    epoch_samples.extend(batch_list)

                verify_order(epoch_samples, dataset_size, batch_size, 2)

        finally:
            loader.close()

    @pytest.mark.parametrize("multithread", [False, True])
    def test_single_consumer_three_workers(self, multithread):
        """Test: 1 consumer, 3 workers, shuffle=False"""
        dataset_size = 47  # 47/3 = 15.67 -> workers get 16, 16, 15
        batch_size = 4
        num_epochs = 5

        loader = SwiftLoader(
            dataset_class=IndexDataset,
            dataset_kwargs={"size": dataset_size},
            batch_size=batch_size,
            nb_consumer=1,
            worker_per_consumer=3,
            shuffle=False,
            multithread=multithread,
            data_queue_size=10,
        )
        loader.start(consumer_index=0)

        try:
            for epoch in range(num_epochs):
                epoch_samples = []

                for batch in loader:
                    batch_list = extract_samples_from_batch(batch)
                    epoch_samples.extend(batch_list)

                verify_order(epoch_samples, dataset_size, batch_size, 3)

        finally:
            loader.close()

    @pytest.mark.parametrize("multithread", [False, True])
    def test_single_consumer_four_workers(self, multithread):
        """Test: 1 consumer, 4 workers, shuffle=False"""
        dataset_size = 97  # Prime: 97/4 = 24.25 -> workers get 25, 24, 24, 24
        batch_size = 4
        num_epochs = 5

        loader = SwiftLoader(
            dataset_class=IndexDataset,
            dataset_kwargs={"size": dataset_size},
            batch_size=batch_size,
            nb_consumer=1,
            worker_per_consumer=4,
            shuffle=False,
            multithread=multithread,
            data_queue_size=10,
        )
        loader.start(consumer_index=0)

        try:
            for epoch in range(num_epochs):
                epoch_samples = []

                for batch in loader:
                    batch_list = extract_samples_from_batch(batch)
                    epoch_samples.extend(batch_list)

                verify_order(epoch_samples, dataset_size, batch_size, 4)

        finally:
            loader.close()

    @pytest.mark.parametrize("multithread", [False, True])
    def test_uneven_dataset_three_workers(self, multithread):
        """Test: Uneven (prime size), 3 workers, shuffle=False"""
        dataset_size = 53  # Prime: 53/3 = 17.67 -> workers get 18, 18, 17
        batch_size = 5
        num_epochs = 5

        loader = SwiftLoader(
            dataset_class=IndexDataset,
            dataset_kwargs={"size": dataset_size},
            batch_size=batch_size,
            nb_consumer=1,
            worker_per_consumer=3,
            shuffle=False,
            multithread=multithread,
            data_queue_size=10,
        )
        loader.start(consumer_index=0)

        try:
            for epoch in range(num_epochs):
                epoch_samples = []

                for batch in loader:
                    batch_list = extract_samples_from_batch(batch)
                    epoch_samples.extend(batch_list)

                verify_order(epoch_samples, dataset_size, batch_size, 3)

        finally:
            loader.close()

    @pytest.mark.parametrize("multithread", [False, True])
    def test_uneven_dataset_four_workers(self, multithread):
        """Test: Uneven (prime size), 4 workers, shuffle=False"""
        dataset_size = 101  # Prime: 101/4 = 25.25 -> workers get 26, 25, 25, 25
        batch_size = 7
        num_epochs = 5

        loader = SwiftLoader(
            dataset_class=IndexDataset,
            dataset_kwargs={"size": dataset_size},
            batch_size=batch_size,
            nb_consumer=1,
            worker_per_consumer=4,
            shuffle=False,
            multithread=multithread,
            data_queue_size=10,
        )
        loader.start(consumer_index=0)

        try:
            for epoch in range(num_epochs):
                epoch_samples = []

                for batch in loader:
                    batch_list = extract_samples_from_batch(batch)
                    epoch_samples.extend(batch_list)

                verify_order(epoch_samples, dataset_size, batch_size, 4)

        finally:
            loader.close()

    @pytest.mark.parametrize("multithread", [False, True])
    def test_uneven_dataset_five_workers(self, multithread):
        """Test: Uneven, 5 workers, shuffle=False"""
        dataset_size = 73  # Prime: 73/5 = 14.6 -> workers get 15, 15, 15, 14, 14
        batch_size = 6
        num_epochs = 5

        loader = SwiftLoader(
            dataset_class=IndexDataset,
            dataset_kwargs={"size": dataset_size},
            batch_size=batch_size,
            nb_consumer=1,
            worker_per_consumer=5,
            shuffle=False,
            multithread=multithread,
            data_queue_size=10,
        )
        loader.start(consumer_index=0)

        try:
            for epoch in range(num_epochs):
                epoch_samples = []

                for batch in loader:
                    batch_list = extract_samples_from_batch(batch)
                    epoch_samples.extend(batch_list)

                verify_order(epoch_samples, dataset_size, batch_size, 5)

        finally:
            loader.close()

    @pytest.mark.parametrize("multithread", [False, True])
    def test_batch_size_one(self, multithread):
        """Test: batch_size=1, 2 workers, shuffle=False"""
        dataset_size = 23  # Prime: 23/2 = 11.5 -> workers get 12, 11
        batch_size = 1
        num_epochs = 5

        loader = SwiftLoader(
            dataset_class=IndexDataset,
            dataset_kwargs={"size": dataset_size},
            batch_size=batch_size,
            nb_consumer=1,
            worker_per_consumer=2,
            shuffle=False,
            multithread=multithread,
            data_queue_size=10,
        )
        loader.start(consumer_index=0)

        try:
            for epoch in range(num_epochs):
                epoch_samples = []

                for batch in loader:
                    batch_list = extract_samples_from_batch(batch)
                    epoch_samples.extend(batch_list)

                verify_order(epoch_samples, dataset_size, batch_size, 2)

        finally:
            loader.close()

    @pytest.mark.parametrize("multithread", [False, True])
    def test_two_consumers_one_worker_each(self, multithread):
        """Test: 2 consumers, 1 worker each, shuffle=False"""
        dataset_size = 50  # 50/2 = 25 per consumer
        batch_size = 5
        num_epochs = 5

        loaders = []
        try:
            for consumer_idx in range(2):
                loader = SwiftLoader(
                    dataset_class=IndexDataset,
                    dataset_kwargs={"size": dataset_size},
                    batch_size=batch_size,
                    nb_consumer=2,
                    worker_per_consumer=1,
                    shuffle=False,
                    multithread=multithread,
                    data_queue_size=10,
                )
                loader.start(consumer_index=consumer_idx)
                loaders.append(loader)

            for epoch in range(num_epochs):
                epoch_consumer_samples = []

                for consumer_idx, loader in enumerate(loaders):
                    consumer_samples = []

                    for batch in loader:
                        batch_list = extract_samples_from_batch(batch)
                        consumer_samples.extend(batch_list)

                    epoch_consumer_samples.append(consumer_samples)

                verify_consumer_order(epoch_consumer_samples, dataset_size, 2)

        finally:
            for loader in loaders:
                loader.close()

    @pytest.mark.parametrize("multithread", [False, True])
    def test_two_consumers_two_workers_each(self, multithread):
        """Test: 2 consumers, 2 workers each, shuffle=False"""
        dataset_size = 50  # 50/2 = 25 per consumer
        batch_size = 4
        num_epochs = 5

        loaders = []
        try:
            for consumer_idx in range(2):
                loader = SwiftLoader(
                    dataset_class=IndexDataset,
                    dataset_kwargs={"size": dataset_size},
                    batch_size=batch_size,
                    nb_consumer=2,
                    worker_per_consumer=2,
                    shuffle=False,
                    multithread=multithread,
                    data_queue_size=10,
                )
                loader.start(consumer_index=consumer_idx)
                loaders.append(loader)

            for epoch in range(num_epochs):
                epoch_consumer_samples = []

                for consumer_idx, loader in enumerate(loaders):
                    consumer_samples = []

                    for batch in loader:
                        batch_list = extract_samples_from_batch(batch)
                        consumer_samples.extend(batch_list)

                    epoch_consumer_samples.append(consumer_samples)

                verify_consumer_order(epoch_consumer_samples, dataset_size, 2)

        finally:
            for loader in loaders:
                loader.close()

    @pytest.mark.parametrize("multithread", [False, True])
    def test_three_consumers_one_worker_each(self, multithread):
        """Test: 3 consumers, 1 worker each, shuffle=False"""
        dataset_size = 47  # 47/3 = 16, 16, 15 per consumer
        batch_size = 5
        num_epochs = 5

        loaders = []
        try:
            for consumer_idx in range(3):
                loader = SwiftLoader(
                    dataset_class=IndexDataset,
                    dataset_kwargs={"size": dataset_size},
                    batch_size=batch_size,
                    nb_consumer=3,
                    worker_per_consumer=1,
                    shuffle=False,
                    multithread=multithread,
                    data_queue_size=10,
                )
                loader.start(consumer_index=consumer_idx)
                loaders.append(loader)

            for epoch in range(num_epochs):
                epoch_consumer_samples = []

                for consumer_idx, loader in enumerate(loaders):
                    consumer_samples = []

                    for batch in loader:
                        batch_list = extract_samples_from_batch(batch)
                        consumer_samples.extend(batch_list)

                    epoch_consumer_samples.append(consumer_samples)

                verify_consumer_order(epoch_consumer_samples, dataset_size, 3)

        finally:
            for loader in loaders:
                loader.close()

    @pytest.mark.parametrize("multithread", [False, True])
    def test_three_consumers_two_workers_each(self, multithread):
        """Test: 3 consumers, 2 workers each, shuffle=False"""
        dataset_size = 47  # 47/3 = 16, 16, 15 per consumer
        batch_size = 4
        num_epochs = 5

        loaders = []
        try:
            for consumer_idx in range(3):
                loader = SwiftLoader(
                    dataset_class=IndexDataset,
                    dataset_kwargs={"size": dataset_size},
                    batch_size=batch_size,
                    nb_consumer=3,
                    worker_per_consumer=2,
                    shuffle=False,
                    multithread=multithread,
                    data_queue_size=10,
                )
                loader.start(consumer_index=consumer_idx)
                loaders.append(loader)

            for epoch in range(num_epochs):
                epoch_consumer_samples = []

                for consumer_idx, loader in enumerate(loaders):
                    consumer_samples = []

                    for batch in loader:
                        batch_list = extract_samples_from_batch(batch)
                        consumer_samples.extend(batch_list)

                    epoch_consumer_samples.append(consumer_samples)

                verify_consumer_order(epoch_consumer_samples, dataset_size, 3)

        finally:
            for loader in loaders:
                loader.close()

    @pytest.mark.parametrize("multithread", [False, True])
    def test_four_consumers_one_worker_each(self, multithread):
        """Test: 4 consumers, 1 worker each, shuffle=False"""
        dataset_size = 97  # 97/4 = 25, 24, 24, 24 per consumer
        batch_size = 5
        num_epochs = 5

        loaders = []
        try:
            for consumer_idx in range(4):
                loader = SwiftLoader(
                    dataset_class=IndexDataset,
                    dataset_kwargs={"size": dataset_size},
                    batch_size=batch_size,
                    nb_consumer=4,
                    worker_per_consumer=1,
                    shuffle=False,
                    multithread=multithread,
                    data_queue_size=10,
                )
                loader.start(consumer_index=consumer_idx)
                loaders.append(loader)

            for epoch in range(num_epochs):
                epoch_consumer_samples = []

                for consumer_idx, loader in enumerate(loaders):
                    consumer_samples = []

                    for batch in loader:
                        batch_list = extract_samples_from_batch(batch)
                        consumer_samples.extend(batch_list)

                    epoch_consumer_samples.append(consumer_samples)

                verify_consumer_order(epoch_consumer_samples, dataset_size, 4)

        finally:
            for loader in loaders:
                loader.close()

    @pytest.mark.parametrize("multithread", [False, True])
    def test_four_consumers_two_workers_each(self, multithread):
        """Test: 4 consumers, 2 workers each, shuffle=False"""
        dataset_size = 97  # 97/4 = 25, 24, 24, 24 per consumer
        batch_size = 4
        num_epochs = 5

        loaders = []
        try:
            for consumer_idx in range(4):
                loader = SwiftLoader(
                    dataset_class=IndexDataset,
                    dataset_kwargs={"size": dataset_size},
                    batch_size=batch_size,
                    nb_consumer=4,
                    worker_per_consumer=2,
                    shuffle=False,
                    multithread=multithread,
                    data_queue_size=10,
                )
                loader.start(consumer_index=consumer_idx)
                loaders.append(loader)

            for epoch in range(num_epochs):
                epoch_consumer_samples = []

                for consumer_idx, loader in enumerate(loaders):
                    consumer_samples = []

                    for batch in loader:
                        batch_list = extract_samples_from_batch(batch)
                        consumer_samples.extend(batch_list)

                    epoch_consumer_samples.append(consumer_samples)

                verify_consumer_order(epoch_consumer_samples, dataset_size, 4)

        finally:
            for loader in loaders:
                loader.close()

    @pytest.mark.parametrize("multithread", [False, True])
    def test_two_consumers_three_workers_each(self, multithread):
        """Test: 2 consumers, 3 workers each, shuffle=False"""
        dataset_size = 50  # 50/2 = 25 per consumer
        batch_size = 3
        num_epochs = 5

        loaders = []
        try:
            for consumer_idx in range(2):
                loader = SwiftLoader(
                    dataset_class=IndexDataset,
                    dataset_kwargs={"size": dataset_size},
                    batch_size=batch_size,
                    nb_consumer=2,
                    worker_per_consumer=3,
                    shuffle=False,
                    multithread=multithread,
                    data_queue_size=10,
                )
                loader.start(consumer_index=consumer_idx)
                loaders.append(loader)

            for epoch in range(num_epochs):
                epoch_consumer_samples = []

                for consumer_idx, loader in enumerate(loaders):
                    consumer_samples = []

                    for batch in loader:
                        batch_list = extract_samples_from_batch(batch)
                        consumer_samples.extend(batch_list)

                    epoch_consumer_samples.append(consumer_samples)

                verify_consumer_order(epoch_consumer_samples, dataset_size, 2)

        finally:
            for loader in loaders:
                loader.close()

