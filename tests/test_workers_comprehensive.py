"""
Comprehensive tests for swift_loader workers

Tests cover:
- Single consumer, single worker
- Single consumer, multiple workers
- Multiple consumers, single worker
- Multiple consumers, multiple workers
- Uneven dataset sizes
- shuffle=True and shuffle=False
- Multiple epochs
- Rotation when shuffle=True
"""

import pytest
import numpy as np
import tempfile
import os
import signal
from contextlib import contextmanager
from typing import List, Tuple, Any
from swift_loader import SwiftLoader

# Try to import pytest-timeout, fallback to signal-based timeout
try:
    import pytest_timeout

    HAS_PYTEST_TIMEOUT = True
except ImportError:
    HAS_PYTEST_TIMEOUT = False

    @contextmanager
    def timeout_context(seconds):
        """Context manager for timeout (Unix only - uses SIGALRM)"""

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Test timed out after {seconds} seconds")

        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


def with_timeout(timeout_seconds=30):
    """Decorator to add timeout to test functions"""
    import functools

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not HAS_PYTEST_TIMEOUT:
                with timeout_context(timeout_seconds):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


class SimpleDataset:
    """Simple test dataset that returns index as data"""

    def __init__(self, size: int):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int) -> Tuple[int, int]:
        if idx >= self.size or idx < 0:
            raise IndexError(
                f"Index {idx} out of range for dataset of size {self.size}"
            )
        return (idx, idx * 2)  # Return tuple (index, index*2)


class IndexDataset:
    """Dataset that returns just the index for easier tracking"""

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


def collect_all_samples(loader: SwiftLoader, num_epochs: int) -> List[List[int]]:
    """Collect all samples from loader across multiple epochs"""
    all_samples = []
    for epoch in range(num_epochs):
        epoch_samples = []
        for batch in loader:
            # Batch will be a numpy array or tensor after collate_fn
            # Convert to list of integers
            if hasattr(batch, "tolist"):
                batch_list = batch.tolist()
            elif isinstance(batch, (list, tuple)):
                batch_list = list(batch)
            else:
                # Try to convert to list
                batch_list = [int(x) for x in batch]

            # Flatten if nested
            if batch_list and isinstance(batch_list[0], (list, tuple)):
                batch_list = [item for sublist in batch_list for item in sublist]

            epoch_samples.extend(batch_list)

        all_samples.append(epoch_samples)
    return all_samples


def verify_epoch_coverage(samples: List[int], dataset_size: int, shuffle: bool):
    """Verify that all samples in dataset are covered in an epoch"""
    unique_samples = set(samples)
    expected_samples = set(range(dataset_size))

    if shuffle:
        # With shuffle, we should see all samples but order may differ
        assert (
            unique_samples == expected_samples
        ), f"Missing samples: {expected_samples - unique_samples}, Extra samples: {unique_samples - expected_samples}"
    else:
        # Without shuffle, order should be preserved
        assert (
            unique_samples == expected_samples
        ), f"Missing samples: {expected_samples - unique_samples}, Extra samples: {unique_samples - expected_samples}"
        # Check order is preserved (within batches)
        # Note: order might be interleaved across workers, so we just check all are present


def verify_rotation(
    samples_per_epoch: List[List[int]], dataset_size: int, nb_worker: int
):
    """Verify that rotation works - epochs should have different sample orders"""
    # With rotation, each epoch should have all samples but in different orders
    # (because workers rotate and get different segments)

    # First verify all epochs have all samples
    for epoch_samples in samples_per_epoch:
        assert set(epoch_samples) == set(
            range(dataset_size)
        ), f"Not all samples present in epoch. Expected {dataset_size}, got {len(set(epoch_samples))}"

    # Check that at least some epochs have different orders
    # Rotation means the order of samples should change across epochs
    orders = [tuple(epoch_samples) for epoch_samples in samples_per_epoch]
    unique_orders = set(orders)

    # With rotation, we should see at least 2 different orders
    # (more if we rotate multiple times and dataset is large enough)
    assert len(unique_orders) >= 2, (
        f"Rotation not working: all epochs have same sample order. "
        f"Unique orders: {len(unique_orders)} out of {len(samples_per_epoch)} epochs"
    )


class TestSingleConsumerSingleWorker:
    """Tests for single consumer, single worker configuration"""

    @pytest.mark.parametrize("shuffle", [True, False])
    @pytest.mark.parametrize(
        "dataset_size", [10, 17, 100, 101]
    )  # Test even and odd sizes
    @pytest.mark.parametrize("batch_size", [1, 3, 5, 10])
    @with_timeout(30)
    def test_single_consumer_single_worker_shuffle(
        self, shuffle, dataset_size, batch_size
    ):
        """Test single consumer, single worker with shuffle on/off"""
        if dataset_size < batch_size:
            pytest.skip("Dataset size must be >= batch_size")

        loader = SwiftLoader(
            dataset_class=IndexDataset,
            dataset_kwargs={"size": dataset_size},
            batch_size=batch_size,
            nb_consumer=1,
            worker_per_consumer=1,
            shuffle=shuffle,
            multithread=True,
            data_queue_size=5,
        )

        loader.start(consumer_index=0)

        # Test 3 epochs
        all_samples = collect_all_samples(loader, num_epochs=3)

        # Verify each epoch covers all samples
        for epoch_samples in all_samples:
            verify_epoch_coverage(epoch_samples, dataset_size, shuffle)

        loader.close()

    @with_timeout(30)
    def test_single_consumer_single_worker_multiple_epochs(self):
        """Test single consumer, single worker across many epochs"""
        dataset_size = 20
        batch_size = 4

        loader = SwiftLoader(
            dataset_class=IndexDataset,
            dataset_kwargs={"size": dataset_size},
            batch_size=batch_size,
            nb_consumer=1,
            worker_per_consumer=1,
            shuffle=True,
            multithread=True,
        )

        loader.start(consumer_index=0)

        # Test 5 epochs
        all_samples = collect_all_samples(loader, num_epochs=5)

        # Verify each epoch
        for epoch_samples in all_samples:
            assert len(set(epoch_samples)) == dataset_size
            assert len(epoch_samples) == dataset_size

        loader.close()


class TestSingleConsumerMultipleWorkers:
    """Tests for single consumer, multiple workers configuration"""

    @pytest.mark.parametrize("shuffle", [True, False])
    @pytest.mark.parametrize("num_workers", [2, 3, 4])
    @pytest.mark.parametrize("dataset_size", [20, 25, 50, 51])  # Test uneven division
    @with_timeout(60)
    def test_single_consumer_multiple_workers(self, shuffle, num_workers, dataset_size):
        """Test single consumer with multiple workers"""
        batch_size = 5

        loader = SwiftLoader(
            dataset_class=IndexDataset,
            dataset_kwargs={"size": dataset_size},
            batch_size=batch_size,
            nb_consumer=1,
            worker_per_consumer=num_workers,
            shuffle=shuffle,
            multithread=True,
            data_queue_size=10,
        )

        loader.start(consumer_index=0)

        # Test 2 epochs
        all_samples = collect_all_samples(loader, num_epochs=2)

        # Verify each epoch
        for epoch_samples in all_samples:
            verify_epoch_coverage(epoch_samples, dataset_size, shuffle)

        loader.close()

    @pytest.mark.parametrize("shuffle", [True, False])
    @with_timeout(60)
    def test_single_consumer_multiple_workers_uneven(self, shuffle):
        """Test with dataset that doesn't divide evenly"""
        dataset_size = 17  # Prime number, won't divide evenly
        batch_size = 3
        num_workers = 3

        loader = SwiftLoader(
            dataset_class=IndexDataset,
            dataset_kwargs={"size": dataset_size},
            batch_size=batch_size,
            nb_consumer=1,
            worker_per_consumer=num_workers,
            shuffle=shuffle,
            multithread=True,
        )

        loader.start(consumer_index=0)

        # Test 3 epochs
        all_samples = collect_all_samples(loader, num_epochs=3)

        for epoch_samples in all_samples:
            verify_epoch_coverage(epoch_samples, dataset_size, shuffle)

        loader.close()


class TestMultipleConsumersSingleWorker:
    """Tests for multiple consumers, single worker per consumer"""

    @pytest.mark.parametrize("shuffle", [True, False])
    @pytest.mark.parametrize("num_consumers", [2, 3])
    @with_timeout(60)
    def test_multiple_consumers_single_worker(self, shuffle, num_consumers):
        """Test multiple consumers, each with single worker"""
        dataset_size = 30
        batch_size = 5

        loader = SwiftLoader(
            dataset_class=IndexDataset,
            dataset_kwargs={"size": dataset_size},
            batch_size=batch_size,
            nb_consumer=num_consumers,
            worker_per_consumer=1,
            shuffle=shuffle,
            multithread=True,
        )

        # Test each consumer separately
        for consumer_idx in range(num_consumers):
            loader.start(consumer_index=consumer_idx)

            # Each consumer should see a subset of data
            all_samples = collect_all_samples(loader, num_epochs=2)

            # Verify each epoch
            for epoch_samples in all_samples:
                # Each consumer sees a portion of the dataset
                assert len(epoch_samples) > 0
                assert all(0 <= s < dataset_size for s in epoch_samples)

            loader.close()
            # Recreate loader for next consumer
            loader = SwiftLoader(
                dataset_class=IndexDataset,
                dataset_kwargs={"size": dataset_size},
                batch_size=batch_size,
                nb_consumer=num_consumers,
                worker_per_consumer=1,
                shuffle=shuffle,
                multithread=True,
            )


class TestMultipleConsumersMultipleWorkers:
    """Tests for multiple consumers, multiple workers per consumer"""

    @pytest.mark.parametrize("shuffle", [True, False])
    @pytest.mark.parametrize("num_consumers", [2, 3])
    @pytest.mark.parametrize("workers_per_consumer", [2, 3])
    @with_timeout(90)
    def test_multiple_consumers_multiple_workers(
        self, shuffle, num_consumers, workers_per_consumer
    ):
        """Test multiple consumers, each with multiple workers"""
        dataset_size = 60
        batch_size = 4

        loader = SwiftLoader(
            dataset_class=IndexDataset,
            dataset_kwargs={"size": dataset_size},
            batch_size=batch_size,
            nb_consumer=num_consumers,
            worker_per_consumer=workers_per_consumer,
            shuffle=shuffle,
            multithread=True,
            data_queue_size=10,
        )

        # Test each consumer
        for consumer_idx in range(num_consumers):
            loader.start(consumer_index=consumer_idx)

            all_samples = collect_all_samples(loader, num_epochs=2)

            for epoch_samples in all_samples:
                assert len(epoch_samples) > 0
                assert all(0 <= s < dataset_size for s in epoch_samples)

            loader.close()

            # Recreate for next consumer
            loader = SwiftLoader(
                dataset_class=IndexDataset,
                dataset_kwargs={"size": dataset_size},
                batch_size=batch_size,
                nb_consumer=num_consumers,
                worker_per_consumer=workers_per_consumer,
                shuffle=shuffle,
                multithread=True,
                data_queue_size=10,
            )


class TestUnevenDatasetSizes:
    """Tests with dataset sizes that don't divide evenly"""

    @pytest.mark.parametrize("shuffle", [True, False])
    @pytest.mark.parametrize(
        "dataset_size", [7, 13, 17, 23, 29, 31, 37, 41, 43, 47]
    )  # Prime numbers
    @with_timeout(60)
    def test_uneven_dataset_sizes(self, shuffle, dataset_size):
        """Test with prime number dataset sizes (won't divide evenly)"""
        batch_size = 3
        num_workers = 4

        loader = SwiftLoader(
            dataset_class=IndexDataset,
            dataset_kwargs={"size": dataset_size},
            batch_size=batch_size,
            nb_consumer=1,
            worker_per_consumer=num_workers,
            shuffle=shuffle,
            multithread=True,
        )

        loader.start(consumer_index=0)

        # Test 2 epochs
        all_samples = collect_all_samples(loader, num_epochs=2)

        for epoch_samples in all_samples:
            verify_epoch_coverage(epoch_samples, dataset_size, shuffle)

        loader.close()

    @pytest.mark.parametrize("shuffle", [True, False])
    @with_timeout(30)
    def test_very_small_dataset(self, shuffle):
        """Test with very small dataset"""
        dataset_size = 5
        batch_size = 2
        num_workers = 3  # More workers than samples

        loader = SwiftLoader(
            dataset_class=IndexDataset,
            dataset_kwargs={"size": dataset_size},
            batch_size=batch_size,
            nb_consumer=1,
            worker_per_consumer=num_workers,
            shuffle=shuffle,
            multithread=True,
        )

        loader.start(consumer_index=0)

        all_samples = collect_all_samples(loader, num_epochs=2)

        for epoch_samples in all_samples:
            verify_epoch_coverage(epoch_samples, dataset_size, shuffle)

        loader.close()

    @pytest.mark.parametrize("shuffle", [True, False])
    @with_timeout(30)
    def test_large_batch_small_dataset(self, shuffle):
        """Test with batch size larger than some worker segments"""
        dataset_size = 10
        batch_size = 7  # Large batch
        num_workers = 3

        loader = SwiftLoader(
            dataset_class=IndexDataset,
            dataset_kwargs={"size": dataset_size},
            batch_size=batch_size,
            nb_consumer=1,
            worker_per_consumer=num_workers,
            shuffle=shuffle,
            multithread=True,
        )

        loader.start(consumer_index=0)

        all_samples = collect_all_samples(loader, num_epochs=2)

        for epoch_samples in all_samples:
            verify_epoch_coverage(epoch_samples, dataset_size, shuffle)

        loader.close()


class TestRotation:
    """Tests for rotation when shuffle=True"""

    @with_timeout(60)
    def test_rotation_single_worker(self):
        """Test rotation with single worker (should still rotate)"""
        dataset_size = 20
        batch_size = 4

        loader = SwiftLoader(
            dataset_class=IndexDataset,
            dataset_kwargs={"size": dataset_size},
            batch_size=batch_size,
            nb_consumer=1,
            worker_per_consumer=1,
            shuffle=True,
            multithread=True,
        )

        loader.start(consumer_index=0)

        # Collect samples from enough epochs to see rotation
        # With 1 worker and dataset_size=20, rotation happens every epoch
        # We need at least 2 epochs to see rotation
        all_samples = collect_all_samples(loader, num_epochs=3)

        # Verify rotation: samples should be in different order across epochs
        # (or at least not identical)
        sample_sets = [set(epoch_samples) for epoch_samples in all_samples]

        # All epochs should have same samples
        for samples in sample_sets:
            assert samples == set(range(dataset_size))

        # Check that at least some epochs have different orders
        # (rotation means the worker segment changes)
        orders = [tuple(epoch_samples) for epoch_samples in all_samples]
        unique_orders = set(orders)
        assert (
            len(unique_orders) >= 2
        ), "Rotation not working: all epochs have same order"

        loader.close()

    @with_timeout(90)
    def test_rotation_multiple_workers(self):
        """Test rotation with multiple workers"""
        dataset_size = 30
        batch_size = 3
        num_workers = 3

        loader = SwiftLoader(
            dataset_class=IndexDataset,
            dataset_kwargs={"size": dataset_size},
            batch_size=batch_size,
            nb_consumer=1,
            worker_per_consumer=num_workers,
            shuffle=True,
            multithread=True,
        )

        loader.start(consumer_index=0)

        # Collect enough epochs to see rotation
        # With 3 workers, we need at least 3+ epochs to see full rotation cycle
        all_samples = collect_all_samples(loader, num_epochs=5)

        # Verify each epoch covers all samples
        for epoch_samples in all_samples:
            verify_epoch_coverage(epoch_samples, dataset_size, shuffle=True)

        # Verify rotation by checking that sample distributions change
        verify_rotation(all_samples, dataset_size, num_workers)

        loader.close()

    @with_timeout(120)
    def test_rotation_at_least_twice(self):
        """Test that rotation happens at least twice"""
        dataset_size = 40
        batch_size = 4
        num_workers = 4

        loader = SwiftLoader(
            dataset_class=IndexDataset,
            dataset_kwargs={"size": dataset_size},
            batch_size=batch_size,
            nb_consumer=1,
            worker_per_consumer=num_workers,
            shuffle=True,
            multithread=True,
        )

        loader.start(consumer_index=0)

        # Collect enough epochs to see at least 2 rotations
        # With 4 workers, we need at least 4+ epochs
        # Let's do 6 epochs to ensure we see multiple rotations
        all_samples = collect_all_samples(loader, num_epochs=6)

        # Verify each epoch
        for epoch_samples in all_samples:
            verify_epoch_coverage(epoch_samples, dataset_size, shuffle=True)

        # Verify rotation happened at least twice
        verify_rotation(all_samples, dataset_size, num_workers)

        # More detailed check: count unique distributions
        worker_size = int(np.ceil(dataset_size / num_workers))
        distributions = []
        for epoch_samples in all_samples:
            # Create signature for this epoch's distribution
            signature = tuple(sorted(epoch_samples))
            distributions.append(signature)

        unique_distributions = set(distributions)
        assert (
            len(unique_distributions) >= 2
        ), f"Rotation not working: only {len(unique_distributions)} unique distribution(s) across 6 epochs"

        loader.close()

    @with_timeout(90)
    def test_rotation_with_uneven_dataset(self):
        """Test rotation with uneven dataset size"""
        dataset_size = 37  # Prime number
        batch_size = 5
        num_workers = 3

        loader = SwiftLoader(
            dataset_class=IndexDataset,
            dataset_kwargs={"size": dataset_size},
            batch_size=batch_size,
            nb_consumer=1,
            worker_per_consumer=num_workers,
            shuffle=True,
            multithread=True,
        )

        loader.start(consumer_index=0)

        # Test multiple epochs
        all_samples = collect_all_samples(loader, num_epochs=5)

        for epoch_samples in all_samples:
            verify_epoch_coverage(epoch_samples, dataset_size, shuffle=True)

        verify_rotation(all_samples, dataset_size, num_workers)

        loader.close()


class TestMultipleEpochsCoverage:
    """Tests to ensure all samples are covered across multiple epochs"""

    @pytest.mark.parametrize("shuffle", [True, False])
    @with_timeout(30)
    def test_coverage_single_epoch(self, shuffle):
        """Test that single epoch covers all samples"""
        dataset_size = 25
        batch_size = 5

        loader = SwiftLoader(
            dataset_class=IndexDataset,
            dataset_kwargs={"size": dataset_size},
            batch_size=batch_size,
            nb_consumer=1,
            worker_per_consumer=2,
            shuffle=shuffle,
            multithread=True,
        )

        loader.start(consumer_index=0)

        all_samples = collect_all_samples(loader, num_epochs=1)

        assert len(all_samples) == 1
        verify_epoch_coverage(all_samples[0], dataset_size, shuffle)

        loader.close()

    @pytest.mark.parametrize("shuffle", [True, False])
    @with_timeout(60)
    def test_coverage_multiple_epochs(self, shuffle):
        """Test that multiple epochs each cover all samples"""
        dataset_size = 30
        batch_size = 6
        num_epochs = 5

        loader = SwiftLoader(
            dataset_class=IndexDataset,
            dataset_kwargs={"size": dataset_size},
            batch_size=batch_size,
            nb_consumer=1,
            worker_per_consumer=3,
            shuffle=shuffle,
            multithread=True,
        )

        loader.start(consumer_index=0)

        all_samples = collect_all_samples(loader, num_epochs=num_epochs)

        assert len(all_samples) == num_epochs

        for epoch_samples in all_samples:
            verify_epoch_coverage(epoch_samples, dataset_size, shuffle)

        loader.close()

    @pytest.mark.parametrize("shuffle", [True, False])
    @with_timeout(60)
    def test_no_duplicates_within_epoch(self, shuffle):
        """Test that no samples are duplicated within an epoch"""
        dataset_size = 20
        batch_size = 4

        loader = SwiftLoader(
            dataset_class=IndexDataset,
            dataset_kwargs={"size": dataset_size},
            batch_size=batch_size,
            nb_consumer=1,
            worker_per_consumer=2,
            shuffle=shuffle,
            multithread=True,
        )

        loader.start(consumer_index=0)

        all_samples = collect_all_samples(loader, num_epochs=3)

        for epoch_samples in all_samples:
            # Check no duplicates within epoch
            assert len(epoch_samples) == len(
                set(epoch_samples)
            ), f"Found duplicates in epoch: {epoch_samples}"
            # Check all samples present
            assert set(epoch_samples) == set(range(dataset_size))

        loader.close()


class TestMultithreadFalse:
    """Tests with multithread=False (single orchestration thread)"""

    @pytest.mark.parametrize("shuffle", [True, False])
    @with_timeout(60)
    def test_multithread_false_single_consumer(self, shuffle):
        """Test multithread=False with single consumer, multiple workers"""
        dataset_size = 20
        batch_size = 4

        loader = SwiftLoader(
            dataset_class=IndexDataset,
            dataset_kwargs={"size": dataset_size},
            batch_size=batch_size,
            nb_consumer=1,
            worker_per_consumer=3,
            shuffle=shuffle,
            multithread=False,  # Use single orchestration thread
        )

        loader.start(consumer_index=0)

        all_samples = collect_all_samples(loader, num_epochs=2)

        for epoch_samples in all_samples:
            verify_epoch_coverage(epoch_samples, dataset_size, shuffle)

        loader.close()

    @pytest.mark.parametrize("shuffle", [True, False])
    @with_timeout(60)
    def test_multithread_false_uneven_dataset(self, shuffle):
        """Test multithread=False with uneven dataset"""
        dataset_size = 17
        batch_size = 3

        loader = SwiftLoader(
            dataset_class=IndexDataset,
            dataset_kwargs={"size": dataset_size},
            batch_size=batch_size,
            nb_consumer=1,
            worker_per_consumer=2,
            shuffle=shuffle,
            multithread=False,
        )

        loader.start(consumer_index=0)

        all_samples = collect_all_samples(loader, num_epochs=2)

        for epoch_samples in all_samples:
            verify_epoch_coverage(epoch_samples, dataset_size, shuffle)

        loader.close()


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    @with_timeout(30)
    def test_single_sample_dataset(self):
        """Test with dataset of size 1"""
        loader = SwiftLoader(
            dataset_class=IndexDataset,
            dataset_kwargs={"size": 1},
            batch_size=1,
            nb_consumer=1,
            worker_per_consumer=1,
            shuffle=True,
            multithread=True,
        )

        loader.start(consumer_index=0)

        all_samples = collect_all_samples(loader, num_epochs=3)

        for epoch_samples in all_samples:
            assert epoch_samples == [0]

        loader.close()

    @with_timeout(30)
    def test_batch_size_one(self):
        """Test with batch size of 1"""
        dataset_size = 10

        loader = SwiftLoader(
            dataset_class=IndexDataset,
            dataset_kwargs={"size": dataset_size},
            batch_size=1,
            nb_consumer=1,
            worker_per_consumer=2,
            shuffle=True,
            multithread=True,
        )

        loader.start(consumer_index=0)

        all_samples = collect_all_samples(loader, num_epochs=2)

        for epoch_samples in all_samples:
            verify_epoch_coverage(epoch_samples, dataset_size, shuffle=True)

        loader.close()

    @with_timeout(30)
    def test_batch_size_equals_dataset(self):
        """Test with batch size equal to dataset size"""
        dataset_size = 10

        loader = SwiftLoader(
            dataset_class=IndexDataset,
            dataset_kwargs={"size": dataset_size},
            batch_size=dataset_size,
            nb_consumer=1,
            worker_per_consumer=1,
            shuffle=True,
            multithread=True,
        )

        loader.start(consumer_index=0)

        all_samples = collect_all_samples(loader, num_epochs=2)

        for epoch_samples in all_samples:
            assert len(epoch_samples) == dataset_size
            assert set(epoch_samples) == set(range(dataset_size))

        loader.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
