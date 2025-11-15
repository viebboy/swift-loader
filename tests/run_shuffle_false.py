"""
Test script specifically for shuffle=False configurations

This script tests shuffle=False to ensure:
1. Data is returned in the correct order (no shuffling)
2. All samples are covered
3. No deadlocks occur

Usage:
    python tests/test_script_shuffle_false.py
    python tests/test_script_shuffle_false.py --multithread
    python tests/test_script_shuffle_false.py --no-multithread
"""

import numpy as np
from swift_loader import SwiftLoader
import time
import sys
import signal
import argparse
from contextlib import contextmanager

# Check if numpy is available
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

# Try to import tqdm, fallback to simple progress if not available
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    # Simple progress bar fallback
    class tqdm:
        def __init__(self, desc="", total=0, unit="", ncols=100, **kwargs):
            self.desc = desc
            self.total = total
            self.unit = unit
            self.current = 0

        def update(self, n=1):
            self.current += n
            if self.total > 0:
                percent = (self.current / self.total) * 100
                bar_length = 50
                filled = int(bar_length * self.current / self.total)
                bar = "=" * filled + "-" * (bar_length - filled)
                sys.stdout.write(
                    f"\r{self.desc} [{bar}] {self.current}/{self.total} {self.unit} ({percent:.1f}%)"
                )
                sys.stdout.flush()

        def close(self):
            print()  # New line after progress


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


class RandomNumpyDataset:
    """Dataset that returns random numpy arrays of specified shape"""

    def __init__(self, size: int, shape: tuple = (32, 32, 3), dtype=np.float32):
        self.size = size
        self.shape = shape
        self.dtype = dtype
        # Pre-generate all arrays for consistency
        np.random.seed(42)
        self.data = [np.random.rand(*shape).astype(dtype) for _ in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int) -> np.ndarray:
        if idx >= self.size or idx < 0:
            raise IndexError(
                f"Index {idx} out of range for dataset of size {self.size}"
            )
        return self.data[idx]


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
    import math

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

    print(f"  ✓ Consumer order verified: {nb_consumer} consumers in correct order")


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
        print("  ✓ Order verified: strictly sequential (single worker)")
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

    print(
        f"  ✓ Order verified: {num_batches} batches in correct order "
        f"({nb_workers} workers, interleaved)"
    )


def test_configuration(
    config_name: str,
    dataset_class,
    dataset_kwargs: dict,
    batch_size: int,
    nb_consumer: int,
    worker_per_consumer: int,
    shuffle: bool,
    num_epochs: int = 3,
    multithread: bool = True,
    data_queue_size: int = 10,
    timeout_seconds: int = 60,
    **kwargs,
):
    """Test a specific configuration and verify order"""
    print(f"\n{'='*80}")
    print(f"Testing: {config_name}")
    print(f"{'='*80}")
    print(f"  Dataset: {dataset_class.__name__}")
    print(f"  Dataset size: {dataset_kwargs.get('size', 'N/A')}")
    print(f"  Batch size: {batch_size}")
    print(f"  Consumers: {nb_consumer}, Workers per consumer: {worker_per_consumer}")
    print(f"  Shuffle: {shuffle} (MUST be False)")
    print(f"  Multithread: {multithread}")
    print(f"  Epochs: {num_epochs}")
    print(f"{'='*80}\n")

    if shuffle:
        raise ValueError("This script only tests shuffle=False!")

    # Create loaders for all consumers
    loaders = []
    try:
        # Get dataset size by creating a temporary dataset instance
        temp_dataset = dataset_class(**dataset_kwargs)
        dataset_size = len(temp_dataset)

        # Create a loader for each consumer
        for consumer_idx in range(nb_consumer):
            loader = SwiftLoader(
                dataset_class=dataset_class,
                dataset_kwargs=dataset_kwargs,
                batch_size=batch_size,
                nb_consumer=nb_consumer,
                worker_per_consumer=worker_per_consumer,
                shuffle=shuffle,
                multithread=multithread,
                data_queue_size=data_queue_size,
                **kwargs,
            )
            loader.start(consumer_index=consumer_idx)
            loaders.append(loader)

        # Calculate expected batches per consumer
        expected_batches_per_consumer = len(loaders[0]) if loaders else 0
        expected_batches_total = expected_batches_per_consumer * nb_consumer

        print(f"Expected batches per consumer: {expected_batches_per_consumer}")
        print(f"Expected batches total (all consumers): {expected_batches_total}")
        print(f"Dataset size: {dataset_size}")

        # Test multiple epochs with timeout
        all_epoch_samples = []
        total_batches = 0
        try:
            with timeout_context(timeout_seconds * num_epochs):
                for epoch in range(num_epochs):
                    epoch_samples = []
                    epoch_consumer_samples = []  # Track samples per consumer
                    epoch_batches = 0

                    # Create progress bar for this epoch
                    pbar = tqdm(
                        desc=f"Epoch {epoch+1}/{num_epochs}",
                        total=expected_batches_total,
                        unit="batch",
                        ncols=100,
                    )

                    try:
                        # Iterate through all consumers to collect all samples
                        for consumer_idx, loader in enumerate(loaders):
                            consumer_samples = []
                            consumer_batches = 0

                            try:
                                for batch in loader:
                                    epoch_batches += 1
                                    total_batches += 1
                                    consumer_batches += 1
                                    pbar.update(1)

                                    # Extract samples from batch
                                    # For IndexDataset, batch contains indices (1D array or list)
                                    # For RandomNumpyDataset, batch is a numpy array of shape (batch_size, *sample_shape)
                                    # We need to handle both cases

                                    if isinstance(batch, np.ndarray):
                                        # For numpy arrays, check if it's from IndexDataset (1D) or RandomNumpyDataset (multi-D)
                                        if batch.ndim == 1:
                                            # 1D array - likely indices from IndexDataset
                                            batch_list = batch.tolist()
                                        else:
                                            # Multi-dimensional array - from RandomNumpyDataset
                                            # The batch size is the first dimension
                                            # We can't extract meaningful sample indices from random data
                                            # So we'll just count the number of samples in this batch
                                            batch_size_actual = batch.shape[0]
                                            # Create sequential indices based on current position
                                            # This allows us to verify we got the right number of samples
                                            batch_list = list(
                                                range(
                                                    len(consumer_samples),
                                                    len(consumer_samples)
                                                    + batch_size_actual,
                                                )
                                            )
                                    elif hasattr(batch, "tolist"):
                                        # Try to convert to list
                                        batch_list = batch.tolist()
                                        # If it's a nested structure, we might need to handle it differently
                                        # But for IndexDataset, it should be a simple list
                                    elif isinstance(batch, (list, tuple)):
                                        batch_list = list(batch)
                                    else:
                                        # Try to convert to integers
                                        try:
                                            batch_list = [int(x) for x in batch]
                                        except (ValueError, TypeError):
                                            # If conversion fails, treat as single sample
                                            batch_list = [batch]

                                    # Flatten if nested (but only if elements are lists/tuples of numbers, not numpy arrays)
                                    # Skip flattening for numpy arrays as they represent the actual data
                                    if (
                                        batch_list
                                        and isinstance(batch_list[0], (list, tuple))
                                        and not isinstance(batch_list[0], np.ndarray)
                                        and not isinstance(batch_list[0], (int, float))
                                    ):
                                        # Only flatten if the nested structure contains numbers, not arrays
                                        try:
                                            batch_list = [
                                                item
                                                for sublist in batch_list
                                                for item in sublist
                                            ]
                                        except (TypeError, ValueError):
                                            # If flattening fails, keep as is
                                            pass

                                    consumer_samples.extend(batch_list)
                                    epoch_samples.extend(batch_list)

                            except StopIteration:
                                pass

                            # Store consumer samples for this epoch
                            epoch_consumer_samples.append(consumer_samples)

                            if nb_consumer > 1:
                                print(
                                    f"    Consumer {consumer_idx}: {consumer_batches} batches, {len(consumer_samples)} samples"
                                )

                    except StopIteration:
                        pass
                    finally:
                        pbar.close()

                    print(
                        f"  Epoch {epoch+1}: {epoch_batches} batches total, "
                        f"{len(epoch_samples)} samples total"
                    )

                    # Verify consumer order if multiple consumers
                    if nb_consumer > 1:
                        verify_consumer_order(
                            epoch_consumer_samples, dataset_size, nb_consumer
                        )

                    # Verify order for this epoch (all consumers combined)
                    verify_order(
                        epoch_samples,
                        dataset_size,
                        batch_size,
                        worker_per_consumer
                        * nb_consumer,  # Total workers across all consumers
                    )

                    all_epoch_samples.append(epoch_samples)

        except TimeoutError as e:
            print(f"\n⚠️  TIMEOUT: {e}")
            print("  This configuration appears to be stuck (likely a deadlock).")
            print(f"  Configuration: {config_name}")
            raise

        print(f"\n✓ Configuration '{config_name}' completed successfully!")
        print(f"  Total batches processed: {total_batches}")
        print(f"  Average batches per epoch: {total_batches / num_epochs:.2f}")
        print(f"  All {num_epochs} epochs verified with correct order!")

    except Exception as e:
        print(f"\n✗ Configuration '{config_name}' FAILED with error:")
        print(f"  {type(e).__name__}: {str(e)}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        # Clean up all loaders
        for loader in loaders:
            if loader is not None:
                try:
                    loader.close()
                except Exception as e:
                    print(f"  Warning: Error during cleanup: {e}")


def main():
    """Run all shuffle=False test configurations"""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Test script for shuffle=False configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/test_script_shuffle_false.py
  python tests/test_script_shuffle_false.py --multithread
  python tests/test_script_shuffle_false.py --no-multithread
        """,
    )
    parser.add_argument(
        "--multithread",
        action="store_true",
        help="Override all test configurations to use multithread=True",
    )
    parser.add_argument(
        "--no-multithread",
        action="store_true",
        help="Override all test configurations to use multithread=False",
    )
    args = parser.parse_args()

    # Determine multithread override
    multithread_override = None
    if args.multithread and args.no_multithread:
        parser.error("Cannot specify both --multithread and --no-multithread")
    elif args.multithread:
        multithread_override = True
    elif args.no_multithread:
        multithread_override = False

    print("\n" + "=" * 80)
    print("Swift Loader Test Script - shuffle=False (Order Verification)")
    print("=" * 80)
    print("\nThis script tests shuffle=False to ensure:")
    print("  1. Data is returned in correct order")
    print("  2. All samples are covered")
    print("  3. No deadlocks occur")
    if multithread_override is not None:
        print(f"  Multithread override: {multithread_override}")
    print("=" * 80)

    test_configs = []

    # ============================================================================
    # Test Case 1: Single consumer, single worker
    # ============================================================================
    test_configs.append(
        {
            "name": "IndexDataset - 1 consumer, 1 worker, shuffle=False",
            "dataset_class": IndexDataset,
            "dataset_kwargs": {"size": 47},  # Prime number (uneven)
            "batch_size": 5,
            "nb_consumer": 1,
            "worker_per_consumer": 1,
            "shuffle": False,
            "multithread": True,
            "num_epochs": 5,  # Longer epochs
        }
    )

    # ============================================================================
    # Test Case 2: Single consumer, multiple workers (all with uneven sizes)
    # ============================================================================
    test_configs.append(
        {
            "name": "IndexDataset - 1 consumer, 3 workers, shuffle=False",
            "dataset_class": IndexDataset,
            "dataset_kwargs": {"size": 47},  # 47/3 = 15.67 -> workers get 16, 16, 15
            "batch_size": 4,
            "nb_consumer": 1,
            "worker_per_consumer": 3,
            "shuffle": False,
            "multithread": True,
            "num_epochs": 5,  # Longer epochs
        }
    )

    test_configs.append(
        {
            "name": "IndexDataset - 1 consumer, 2 workers, shuffle=False",
            "dataset_class": IndexDataset,
            "dataset_kwargs": {"size": 43},  # Prime: 43/2 = 21.5 -> workers get 22, 21
            "batch_size": 5,
            "nb_consumer": 1,
            "worker_per_consumer": 2,
            "shuffle": False,
            "multithread": True,
            "num_epochs": 5,  # Longer epochs
        }
    )

    test_configs.append(
        {
            "name": "IndexDataset - 1 consumer, 4 workers, shuffle=False",
            "dataset_class": IndexDataset,
            "dataset_kwargs": {
                "size": 97
            },  # Prime: 97/4 = 24.25 -> workers get 25, 24, 24, 24
            "batch_size": 4,
            "nb_consumer": 1,
            "worker_per_consumer": 4,
            "shuffle": False,
            "multithread": True,
            "num_epochs": 5,  # Longer epochs
        }
    )

    # ============================================================================
    # Test Case 3: More uneven dataset sizes with different worker counts
    # ============================================================================
    test_configs.append(
        {
            "name": "IndexDataset - Uneven (prime size), 3 workers, shuffle=False",
            "dataset_class": IndexDataset,
            "dataset_kwargs": {
                "size": 53
            },  # Prime: 53/3 = 17.67 -> workers get 18, 18, 17
            "batch_size": 5,
            "nb_consumer": 1,
            "worker_per_consumer": 3,
            "shuffle": False,
            "multithread": True,
            "num_epochs": 5,  # Longer epochs
        }
    )

    test_configs.append(
        {
            "name": "IndexDataset - Uneven (prime size), 4 workers, shuffle=False",
            "dataset_class": IndexDataset,
            "dataset_kwargs": {
                "size": 101
            },  # Prime: 101/4 = 25.25 -> workers get 26, 25, 25, 25
            "batch_size": 7,
            "nb_consumer": 1,
            "worker_per_consumer": 4,
            "shuffle": False,
            "multithread": True,
            "num_epochs": 5,  # Longer epochs
        }
    )

    test_configs.append(
        {
            "name": "IndexDataset - Uneven, 5 workers, shuffle=False",
            "dataset_class": IndexDataset,
            "dataset_kwargs": {
                "size": 73
            },  # Prime: 73/5 = 14.6 -> workers get 15, 15, 15, 14, 14
            "batch_size": 6,
            "nb_consumer": 1,
            "worker_per_consumer": 5,
            "shuffle": False,
            "multithread": True,
            "num_epochs": 5,  # Longer epochs
        }
    )

    # ============================================================================
    # Test Case 4: Multithread=False - uneven size
    # ============================================================================
    test_configs.append(
        {
            "name": "IndexDataset - multithread=False, 3 workers, shuffle=False",
            "dataset_class": IndexDataset,
            "dataset_kwargs": {"size": 47},  # 47/3 = 15.67 -> workers get 16, 16, 15
            "batch_size": 5,
            "nb_consumer": 1,
            "worker_per_consumer": 3,
            "shuffle": False,
            "multithread": False,
            "num_epochs": 5,  # Longer epochs
        }
    )

    # ============================================================================
    # Test Case 5: Small batch size - uneven size
    # ============================================================================
    test_configs.append(
        {
            "name": "IndexDataset - batch_size=1, 2 workers, shuffle=False",
            "dataset_class": IndexDataset,
            "dataset_kwargs": {"size": 23},  # Prime: 23/2 = 11.5 -> workers get 12, 11
            "batch_size": 1,
            "nb_consumer": 1,
            "worker_per_consumer": 2,
            "shuffle": False,
            "multithread": True,
            "num_epochs": 5,  # Longer epochs
        }
    )

    # ============================================================================
    # Test Case 6: Multiple consumers - verify consumer order
    # ============================================================================
    test_configs.append(
        {
            "name": "IndexDataset - 2 consumers, 1 worker each, shuffle=False",
            "dataset_class": IndexDataset,
            "dataset_kwargs": {"size": 50},  # 50/2 = 25 per consumer
            "batch_size": 5,
            "nb_consumer": 2,
            "worker_per_consumer": 1,
            "shuffle": False,
            "multithread": True,
            "num_epochs": 5,
        }
    )

    test_configs.append(
        {
            "name": "IndexDataset - 2 consumers, 2 workers each, shuffle=False",
            "dataset_class": IndexDataset,
            "dataset_kwargs": {"size": 50},  # 50/2 = 25 per consumer
            "batch_size": 4,
            "nb_consumer": 2,
            "worker_per_consumer": 2,
            "shuffle": False,
            "multithread": True,
            "num_epochs": 5,
        }
    )

    test_configs.append(
        {
            "name": "IndexDataset - 3 consumers, 1 worker each, shuffle=False",
            "dataset_class": IndexDataset,
            "dataset_kwargs": {"size": 47},  # 47/3 = 16, 16, 15 per consumer
            "batch_size": 5,
            "nb_consumer": 3,
            "worker_per_consumer": 1,
            "shuffle": False,
            "multithread": True,
            "num_epochs": 5,
        }
    )

    test_configs.append(
        {
            "name": "IndexDataset - 3 consumers, 2 workers each, shuffle=False",
            "dataset_class": IndexDataset,
            "dataset_kwargs": {"size": 47},  # 47/3 = 16, 16, 15 per consumer
            "batch_size": 4,
            "nb_consumer": 3,
            "worker_per_consumer": 2,
            "shuffle": False,
            "multithread": True,
            "num_epochs": 5,
        }
    )

    test_configs.append(
        {
            "name": "IndexDataset - 4 consumers, 1 worker each, shuffle=False",
            "dataset_class": IndexDataset,
            "dataset_kwargs": {"size": 97},  # 97/4 = 25, 24, 24, 24 per consumer
            "batch_size": 5,
            "nb_consumer": 4,
            "worker_per_consumer": 1,
            "shuffle": False,
            "multithread": True,
            "num_epochs": 5,
        }
    )

    test_configs.append(
        {
            "name": "IndexDataset - 4 consumers, 2 workers each, shuffle=False",
            "dataset_class": IndexDataset,
            "dataset_kwargs": {"size": 97},  # 97/4 = 25, 24, 24, 24 per consumer
            "batch_size": 4,
            "nb_consumer": 4,
            "worker_per_consumer": 2,
            "shuffle": False,
            "multithread": True,
            "num_epochs": 5,
        }
    )

    test_configs.append(
        {
            "name": "IndexDataset - 2 consumers, 3 workers each, shuffle=False",
            "dataset_class": IndexDataset,
            "dataset_kwargs": {"size": 50},  # 50/2 = 25 per consumer
            "batch_size": 3,
            "nb_consumer": 2,
            "worker_per_consumer": 3,
            "shuffle": False,
            "multithread": True,
            "num_epochs": 5,
        }
    )

    # ============================================================================
    # Run all tests
    # ============================================================================

    # Apply multithread override if specified
    if multithread_override is not None:
        for config in test_configs:
            config["multithread"] = multithread_override

    print(f"\nTotal test configurations: {len(test_configs)}")
    print("  All tests use shuffle=False")
    if multithread_override is not None:
        print(f"  All tests use multithread={multithread_override} (overridden)")
    print()

    start_time = time.time()
    passed = 0
    failed = 0

    for i, config in enumerate(test_configs, 1):
        print(f"\n[{i}/{len(test_configs)}]")

        try:
            # Extract 'name' and pass it as 'config_name'
            config_copy = config.copy()
            config_name = config_copy.pop("name")
            test_configuration(config_name=config_name, **config_copy)
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n✗ Test {i} failed: {e}")
            # Continue with next test

    elapsed_time = time.time() - start_time

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total configurations tested: {len(test_configs)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(
        f"Average time per configuration: "
        f"{elapsed_time / len(test_configs):.2f} seconds"
    )
    print("=" * 80)

    if failed > 0:
        print(f"\n⚠️  {failed} configuration(s) failed!")
        exit(1)
    else:
        print("\n✓ All configurations passed!")
        print("  ✓ All data returned in correct order")
        print("  ✓ No deadlocks detected")
        exit(0)


if __name__ == "__main__":
    main()
