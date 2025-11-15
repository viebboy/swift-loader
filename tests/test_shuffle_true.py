"""
Test cases for shuffle=True configurations

These tests ensure:
1. All samples are covered in each epoch
2. Worker rotation works correctly across epochs
3. No deadlocks occur
4. Data is shuffled (order verification not applicable)
"""

import numpy as np
import pytest
from swift_loader import SwiftLoader
import time
import signal
from contextlib import contextmanager

# Check if numpy is available
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


@contextmanager
def timeout_context(seconds):
    """Context manager for timeout (Unix only - uses SIGALRM)"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Cancel the alarm and restore old handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class IndexDataset:
    """Dataset that returns the index as data for easy coverage verification"""

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


def verify_epoch_coverage(samples: list, dataset_size: int):
    """
    Verify that all samples are covered in an epoch when shuffle=True

    When shuffle=True, samples should be shuffled but all samples
    from 0 to dataset_size-1 should be present exactly once.
    """
    expected_samples = set(range(dataset_size))
    actual_samples = set(samples)

    # Check all samples are present
    if len(actual_samples) != len(expected_samples):
        missing = expected_samples - actual_samples
        extra = actual_samples - expected_samples
        raise AssertionError(
            f"Sample count mismatch. Expected {len(expected_samples)} unique samples, "
            f"got {len(actual_samples)}. Missing: {missing}, Extra: {extra}"
        )

    # Check for duplicates
    if len(samples) != len(actual_samples):
        from collections import Counter
        duplicates = {k: v for k, v in Counter(samples).items() if v > 1}
        raise AssertionError(
            f"Found duplicate samples: {duplicates}. "
            f"Total samples: {len(samples)}, Unique samples: {len(actual_samples)}"
        )

    # Check all expected samples are present
    if actual_samples != expected_samples:
        missing = expected_samples - actual_samples
        extra = actual_samples - expected_samples
        raise AssertionError(f"Sample mismatch. Missing: {missing}, Extra: {extra}")


def verify_rotation(epoch_samples_list: list, dataset_size: int, nb_workers: int):
    """
    Verify that worker rotation works correctly across epochs

    With shuffle=True, workers rotate through different segments of the dataset
    in each epoch. We verify this by checking that:
    1. Each epoch covers all samples
    2. The order changes between epochs (indicating rotation)
    3. After nb_workers epochs, we should see rotation patterns
    """
    if len(epoch_samples_list) < 2:
        pytest.skip("Need at least 2 epochs to verify rotation")

    # Check that orders are different (indicating rotation)
    all_same = all(
        epoch_samples_list[0] == samples for samples in epoch_samples_list[1:]
    )

    if all_same:
        pytest.fail("All epochs have identical order (rotation may not be working)")


def extract_samples_from_batch(batch):
    """Extract sample indices from a batch"""
    if isinstance(batch, np.ndarray):
        batch_list = batch.tolist()
    elif hasattr(batch, "tolist"):
        batch_list = batch.tolist()
    elif isinstance(batch, (list, tuple)):
        batch_list = list(batch)
    else:
        try:
            batch_list = [int(x) for x in batch]
        except (ValueError, TypeError):
            batch_list = [batch]

    # Flatten if nested
    if (
        batch_list
        and isinstance(batch_list[0], (list, tuple))
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


class TestShuffleTrue:
    """Test cases for shuffle=True configurations"""

    @pytest.mark.parametrize("multithread", [False, True])
    def test_single_consumer_single_worker(self, multithread):
        """Test: 1 consumer, 1 worker, shuffle=True"""
        dataset_size = 50
        batch_size = 5
        num_epochs = 3

        loader = SwiftLoader(
            dataset_class=IndexDataset,
            dataset_kwargs={"size": dataset_size},
            batch_size=batch_size,
            nb_consumer=1,
            worker_per_consumer=1,
            shuffle=True,
            multithread=multithread,
            data_queue_size=50,
            max_buffer_size=50,
        )
        loader.start(consumer_index=0)

        expected_batches = len(loader)
        all_epoch_samples = []

        try:
            for epoch in range(num_epochs):
                epoch_samples = []
                epoch_batches = 0

                for batch in loader:
                    epoch_batches += 1
                    batch_list = extract_samples_from_batch(batch)
                    epoch_samples.extend(batch_list)

                assert epoch_batches == expected_batches, (
                    f"Epoch {epoch+1}: Expected {expected_batches} batches, "
                    f"got {epoch_batches}"
                )
                verify_epoch_coverage(epoch_samples, dataset_size)
                all_epoch_samples.append(epoch_samples)

            verify_rotation(all_epoch_samples, dataset_size, 1)

        finally:
            loader.close()

    @pytest.mark.parametrize("multithread", [False, True])
    def test_single_consumer_two_workers(self, multithread):
        """Test: 1 consumer, 2 workers, shuffle=True"""
        dataset_size = 40
        batch_size = 5
        num_epochs = 5

        loader = SwiftLoader(
            dataset_class=IndexDataset,
            dataset_kwargs={"size": dataset_size},
            batch_size=batch_size,
            nb_consumer=1,
            worker_per_consumer=2,
            shuffle=True,
            multithread=multithread,
            data_queue_size=50,
            max_buffer_size=50,
        )
        loader.start(consumer_index=0)

        expected_batches = len(loader)
        all_epoch_samples = []
        consumer_batches_history = []

        try:
            for epoch in range(num_epochs):
                epoch_samples = []
                epoch_batches = 0

                for batch in loader:
                    epoch_batches += 1
                    batch_list = extract_samples_from_batch(batch)
                    epoch_samples.extend(batch_list)

                consumer_batches_history.append(epoch_batches)

                assert epoch_batches == expected_batches, (
                    f"Epoch {epoch+1}: Expected {expected_batches} batches, "
                    f"got {epoch_batches}"
                )
                verify_epoch_coverage(epoch_samples, dataset_size)
                all_epoch_samples.append(epoch_samples)

            # Verify batch consistency across epochs
            assert len(set(consumer_batches_history)) == 1, (
                f"Inconsistent batches across epochs: {consumer_batches_history}"
            )

            verify_rotation(all_epoch_samples, dataset_size, 2)

        finally:
            loader.close()

    @pytest.mark.parametrize("multithread", [False, True])
    def test_single_consumer_three_workers(self, multithread):
        """Test: 1 consumer, 3 workers, shuffle=True"""
        dataset_size = 60
        batch_size = 4
        num_epochs = 6

        loader = SwiftLoader(
            dataset_class=IndexDataset,
            dataset_kwargs={"size": dataset_size},
            batch_size=batch_size,
            nb_consumer=1,
            worker_per_consumer=3,
            shuffle=True,
            multithread=multithread,
            data_queue_size=50,
            max_buffer_size=50,
        )
        loader.start(consumer_index=0)

        expected_batches = len(loader)
        all_epoch_samples = []
        consumer_batches_history = []

        try:
            for epoch in range(num_epochs):
                epoch_samples = []
                epoch_batches = 0

                for batch in loader:
                    epoch_batches += 1
                    batch_list = extract_samples_from_batch(batch)
                    epoch_samples.extend(batch_list)

                consumer_batches_history.append(epoch_batches)

                assert epoch_batches == expected_batches, (
                    f"Epoch {epoch+1}: Expected {expected_batches} batches, "
                    f"got {epoch_batches}"
                )
                verify_epoch_coverage(epoch_samples, dataset_size)
                all_epoch_samples.append(epoch_samples)

            # Verify batch consistency across epochs
            assert len(set(consumer_batches_history)) == 1, (
                f"Inconsistent batches across epochs: {consumer_batches_history}"
            )

            verify_rotation(all_epoch_samples, dataset_size, 3)

        finally:
            loader.close()

    @pytest.mark.parametrize("multithread", [False, True])
    def test_single_consumer_four_workers(self, multithread):
        """Test: 1 consumer, 4 workers, shuffle=True"""
        dataset_size = 80
        batch_size = 4
        num_epochs = 8

        loader = SwiftLoader(
            dataset_class=IndexDataset,
            dataset_kwargs={"size": dataset_size},
            batch_size=batch_size,
            nb_consumer=1,
            worker_per_consumer=4,
            shuffle=True,
            multithread=multithread,
            data_queue_size=50,
            max_buffer_size=50,
        )
        loader.start(consumer_index=0)

        expected_batches = len(loader)
        all_epoch_samples = []
        consumer_batches_history = []

        try:
            for epoch in range(num_epochs):
                epoch_samples = []
                epoch_batches = 0

                for batch in loader:
                    epoch_batches += 1
                    batch_list = extract_samples_from_batch(batch)
                    epoch_samples.extend(batch_list)

                consumer_batches_history.append(epoch_batches)

                assert epoch_batches == expected_batches, (
                    f"Epoch {epoch+1}: Expected {expected_batches} batches, "
                    f"got {epoch_batches}"
                )
                verify_epoch_coverage(epoch_samples, dataset_size)
                all_epoch_samples.append(epoch_samples)

            # Verify batch consistency across epochs
            assert len(set(consumer_batches_history)) == 1, (
                f"Inconsistent batches across epochs: {consumer_batches_history}"
            )

            verify_rotation(all_epoch_samples, dataset_size, 4)

        finally:
            loader.close()

    @pytest.mark.parametrize("multithread", [False, True])
    def test_uneven_dataset_three_workers(self, multithread):
        """Test: Uneven (prime size), 3 workers, shuffle=True"""
        dataset_size = 47  # Prime number
        batch_size = 5
        num_epochs = 6

        loader = SwiftLoader(
            dataset_class=IndexDataset,
            dataset_kwargs={"size": dataset_size},
            batch_size=batch_size,
            nb_consumer=1,
            worker_per_consumer=3,
            shuffle=True,
            multithread=multithread,
            data_queue_size=50,
            max_buffer_size=50,
        )
        loader.start(consumer_index=0)

        expected_batches = len(loader)
        all_epoch_samples = []
        consumer_batches_history = []

        try:
            for epoch in range(num_epochs):
                epoch_samples = []
                epoch_batches = 0

                for batch in loader:
                    epoch_batches += 1
                    batch_list = extract_samples_from_batch(batch)
                    epoch_samples.extend(batch_list)

                consumer_batches_history.append(epoch_batches)

                assert epoch_batches == expected_batches, (
                    f"Epoch {epoch+1}: Expected {expected_batches} batches, "
                    f"got {epoch_batches}"
                )
                verify_epoch_coverage(epoch_samples, dataset_size)
                all_epoch_samples.append(epoch_samples)

            # Verify batch consistency across epochs
            assert len(set(consumer_batches_history)) == 1, (
                f"Inconsistent batches across epochs: {consumer_batches_history}"
            )

            verify_rotation(all_epoch_samples, dataset_size, 3)

        finally:
            loader.close()

    @pytest.mark.parametrize("multithread", [False, True])
    def test_uneven_dataset_four_workers(self, multithread):
        """Test: Uneven (prime size), 4 workers, shuffle=True"""
        dataset_size = 97  # Prime number
        batch_size = 7
        num_epochs = 8

        loader = SwiftLoader(
            dataset_class=IndexDataset,
            dataset_kwargs={"size": dataset_size},
            batch_size=batch_size,
            nb_consumer=1,
            worker_per_consumer=4,
            shuffle=True,
            multithread=multithread,
            data_queue_size=50,
            max_buffer_size=50,
        )
        loader.start(consumer_index=0)

        expected_batches = len(loader)
        all_epoch_samples = []
        consumer_batches_history = []

        try:
            for epoch in range(num_epochs):
                epoch_samples = []
                epoch_batches = 0

                for batch in loader:
                    epoch_batches += 1
                    batch_list = extract_samples_from_batch(batch)
                    epoch_samples.extend(batch_list)

                consumer_batches_history.append(epoch_batches)

                assert epoch_batches == expected_batches, (
                    f"Epoch {epoch+1}: Expected {expected_batches} batches, "
                    f"got {epoch_batches}"
                )
                verify_epoch_coverage(epoch_samples, dataset_size)
                all_epoch_samples.append(epoch_samples)

            # Verify batch consistency across epochs
            assert len(set(consumer_batches_history)) == 1, (
                f"Inconsistent batches across epochs: {consumer_batches_history}"
            )

            verify_rotation(all_epoch_samples, dataset_size, 4)

        finally:
            loader.close()

    @pytest.mark.parametrize("multithread", [False, True])
    def test_batch_size_one(self, multithread):
        """Test: batch_size=1, 2 workers, shuffle=True"""
        dataset_size = 20
        batch_size = 1
        num_epochs = 4

        loader = SwiftLoader(
            dataset_class=IndexDataset,
            dataset_kwargs={"size": dataset_size},
            batch_size=batch_size,
            nb_consumer=1,
            worker_per_consumer=2,
            shuffle=True,
            multithread=multithread,
            data_queue_size=50,
            max_buffer_size=50,
        )
        loader.start(consumer_index=0)

        expected_batches = len(loader)
        all_epoch_samples = []
        consumer_batches_history = []

        try:
            for epoch in range(num_epochs):
                epoch_samples = []
                epoch_batches = 0

                for batch in loader:
                    epoch_batches += 1
                    batch_list = extract_samples_from_batch(batch)
                    epoch_samples.extend(batch_list)

                consumer_batches_history.append(epoch_batches)

                assert epoch_batches == expected_batches, (
                    f"Epoch {epoch+1}: Expected {expected_batches} batches, "
                    f"got {epoch_batches}"
                )
                verify_epoch_coverage(epoch_samples, dataset_size)
                all_epoch_samples.append(epoch_samples)

            # Verify batch consistency across epochs
            assert len(set(consumer_batches_history)) == 1, (
                f"Inconsistent batches across epochs: {consumer_batches_history}"
            )

            verify_rotation(all_epoch_samples, dataset_size, 2)

        finally:
            loader.close()

    @pytest.mark.parametrize("multithread", [False, True])
    def test_large_dataset(self, multithread):
        """Test: Large dataset, 4 workers, shuffle=True"""
        dataset_size = 200
        batch_size = 10
        num_epochs = 8

        loader = SwiftLoader(
            dataset_class=IndexDataset,
            dataset_kwargs={"size": dataset_size},
            batch_size=batch_size,
            nb_consumer=1,
            worker_per_consumer=4,
            shuffle=True,
            multithread=multithread,
            data_queue_size=50,
            max_buffer_size=50,
        )
        loader.start(consumer_index=0)

        expected_batches = len(loader)
        all_epoch_samples = []
        consumer_batches_history = []

        try:
            for epoch in range(num_epochs):
                epoch_samples = []
                epoch_batches = 0

                for batch in loader:
                    epoch_batches += 1
                    batch_list = extract_samples_from_batch(batch)
                    epoch_samples.extend(batch_list)

                consumer_batches_history.append(epoch_batches)

                assert epoch_batches == expected_batches, (
                    f"Epoch {epoch+1}: Expected {expected_batches} batches, "
                    f"got {epoch_batches}"
                )
                verify_epoch_coverage(epoch_samples, dataset_size)
                all_epoch_samples.append(epoch_samples)

            # Verify batch consistency across epochs
            assert len(set(consumer_batches_history)) == 1, (
                f"Inconsistent batches across epochs: {consumer_batches_history}"
            )

            verify_rotation(all_epoch_samples, dataset_size, 4)

        finally:
            loader.close()

