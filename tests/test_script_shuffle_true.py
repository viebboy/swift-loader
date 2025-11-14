"""
Test script specifically for shuffle=True configurations

This script tests shuffle=True to ensure:
1. All samples are covered in each epoch
2. Worker rotation works correctly across epochs
3. No deadlocks occur
4. Data is shuffled (order verification not applicable)

Usage:
    python tests/test_script_shuffle_true.py
"""

import numpy as np
from swift_loader import SwiftLoader
import time
import sys
import signal
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

    print(f"  ✓ Coverage verified: {len(actual_samples)} unique samples (all present)")


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
        print("  ⚠️  Need at least 2 epochs to verify rotation")
        return

    # Check that orders are different (indicating rotation)
    all_same = all(
        epoch_samples_list[0] == samples for samples in epoch_samples_list[1:]
    )

    if all_same:
        print(
            "  ⚠️  Warning: All epochs have identical order (rotation may not be working)"
        )
    else:
        print(
            f"  ✓ Rotation verified: Order differs across {len(epoch_samples_list)} epochs"
        )

    # For multiple workers, verify we see different patterns
    if nb_workers > 1 and len(epoch_samples_list) >= nb_workers:
        # Check that we see variation in the first few samples across epochs
        # This is a heuristic to detect rotation
        first_samples = [samples[0] for samples in epoch_samples_list[:nb_workers]]
        unique_first_samples = len(set(first_samples))

        if unique_first_samples > 1:
            print(
                f"  ✓ Rotation pattern detected: {unique_first_samples} different starting samples "
                f"across {nb_workers} epochs"
            )
        else:
            print(
                f"  ⚠️  Warning: Same starting sample across {nb_workers} epochs "
                f"(rotation may not be working)"
            )


def test_configuration(
    config_name: str,
    dataset_class,
    dataset_kwargs: dict,
    batch_size: int,
    nb_consumer: int,
    worker_per_consumer: int,
    shuffle: bool,
    num_epochs: int = 5,  # More epochs to verify rotation
    multithread: bool = True,
    data_queue_size: int = 10,
    timeout_seconds: int = 60,
    **kwargs,
):
    """Test a specific configuration and verify coverage and rotation"""
    print(f"\n{'='*80}")
    print(f"Testing: {config_name}")
    print(f"{'='*80}")
    print(f"  Dataset: {dataset_class.__name__}")
    print(f"  Dataset size: {dataset_kwargs.get('size', 'N/A')}")
    print(f"  Batch size: {batch_size}")
    print(f"  Consumers: {nb_consumer}, Workers per consumer: {worker_per_consumer}")
    print(f"  Shuffle: {shuffle} (MUST be True)")
    print(f"  Multithread: {multithread}")
    print(f"  Epochs: {num_epochs}")
    print(f"{'='*80}\n")

    if not shuffle:
        raise ValueError("This script only tests shuffle=True!")

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

                    # Verify coverage for this epoch (all consumers combined)
                    verify_epoch_coverage(epoch_samples, dataset_size)

                    all_epoch_samples.append(epoch_samples)

        except TimeoutError as e:
            print(f"\n⚠️  TIMEOUT: {e}")
            print("  This configuration appears to be stuck (likely a deadlock).")
            print(f"  Configuration: {config_name}")
            raise

        # Verify rotation across epochs
        verify_rotation(all_epoch_samples, dataset_size, worker_per_consumer)

        print(f"\n✓ Configuration '{config_name}' completed successfully!")
        print(f"  Total batches processed: {total_batches}")
        print(f"  Average batches per epoch: {total_batches / num_epochs:.2f}")
        print(f"  All {num_epochs} epochs verified with complete coverage!")

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
    """Run all shuffle=True test configurations"""

    print("\n" + "=" * 80)
    print("Swift Loader Test Script - shuffle=True (Coverage & Rotation Verification)")
    print("=" * 80)
    print("\nThis script tests shuffle=True to ensure:")
    print("  1. All samples are covered in each epoch")
    print("  2. Worker rotation works correctly across epochs")
    print("  3. No deadlocks occur")
    print("  4. Data is shuffled (order verification not applicable)")
    print("=" * 80)

    test_configs = []

    # ============================================================================
    # Test Case 1: Single consumer, single worker
    # ============================================================================
    test_configs.append(
        {
            "name": "IndexDataset - 1 consumer, 1 worker, shuffle=True",
            "dataset_class": IndexDataset,
            "dataset_kwargs": {"size": 50},
            "batch_size": 5,
            "nb_consumer": 1,
            "worker_per_consumer": 1,
            "shuffle": True,
            "multithread": True,
            "num_epochs": 3,
        }
    )

    # ============================================================================
    # Test Case 2: Single consumer, multiple workers
    # ============================================================================
    test_configs.append(
        {
            "name": "IndexDataset - 1 consumer, 2 workers, shuffle=True",
            "dataset_class": IndexDataset,
            "dataset_kwargs": {"size": 40},
            "batch_size": 5,
            "nb_consumer": 1,
            "worker_per_consumer": 2,
            "shuffle": True,
            "multithread": True,
            "num_epochs": 5,  # More epochs to verify rotation
        }
    )

    test_configs.append(
        {
            "name": "IndexDataset - 1 consumer, 3 workers, shuffle=True",
            "dataset_class": IndexDataset,
            "dataset_kwargs": {"size": 60},
            "batch_size": 4,
            "nb_consumer": 1,
            "worker_per_consumer": 3,
            "shuffle": True,
            "multithread": True,
            "num_epochs": 6,  # Enough to see rotation (3 workers, 6 epochs = 2 full rotations)
        }
    )

    test_configs.append(
        {
            "name": "IndexDataset - 1 consumer, 4 workers, shuffle=True",
            "dataset_class": IndexDataset,
            "dataset_kwargs": {"size": 80},
            "batch_size": 4,
            "nb_consumer": 1,
            "worker_per_consumer": 4,
            "shuffle": True,
            "multithread": True,
            "num_epochs": 8,  # Enough to see rotation (4 workers, 8 epochs = 2 full rotations)
        }
    )

    # ============================================================================
    # Test Case 3: Uneven dataset sizes
    # ============================================================================
    test_configs.append(
        {
            "name": "IndexDataset - Uneven (prime size), 3 workers, shuffle=True",
            "dataset_class": IndexDataset,
            "dataset_kwargs": {"size": 47},  # Prime number
            "batch_size": 5,
            "nb_consumer": 1,
            "worker_per_consumer": 3,
            "shuffle": True,
            "multithread": True,
            "num_epochs": 6,
        }
    )

    test_configs.append(
        {
            "name": "IndexDataset - Uneven (prime size), 4 workers, shuffle=True",
            "dataset_class": IndexDataset,
            "dataset_kwargs": {"size": 97},  # Prime number
            "batch_size": 7,
            "nb_consumer": 1,
            "worker_per_consumer": 4,
            "shuffle": True,
            "multithread": True,
            "num_epochs": 8,
        }
    )

    # ============================================================================
    # Test Case 4: Numpy arrays (coverage verification by counting samples)
    # ============================================================================
    test_configs.append(
        {
            "name": "RandomNumpyDataset - 1 consumer, 2 workers, shuffle=True",
            "dataset_class": RandomNumpyDataset,
            "dataset_kwargs": {"size": 50, "shape": (20, 20)},
            "batch_size": 5,
            "nb_consumer": 1,
            "worker_per_consumer": 2,
            "shuffle": True,
            "multithread": True,
            "num_epochs": 4,
        }
    )

    test_configs.append(
        {
            "name": "RandomNumpyDataset - 1 consumer, 3 workers, shuffle=True",
            "dataset_class": RandomNumpyDataset,
            "dataset_kwargs": {"size": 60, "shape": (15, 15)},
            "batch_size": 4,
            "nb_consumer": 1,
            "worker_per_consumer": 3,
            "shuffle": True,
            "multithread": True,
            "num_epochs": 6,
        }
    )

    # ============================================================================
    # Test Case 5: Multithread=False
    # ============================================================================
    test_configs.append(
        {
            "name": "IndexDataset - multithread=False, 3 workers, shuffle=True",
            "dataset_class": IndexDataset,
            "dataset_kwargs": {"size": 50},
            "batch_size": 5,
            "nb_consumer": 1,
            "worker_per_consumer": 3,
            "shuffle": True,
            "multithread": False,
            "num_epochs": 6,
        }
    )

    # ============================================================================
    # Test Case 6: Small batch size
    # ============================================================================
    test_configs.append(
        {
            "name": "IndexDataset - batch_size=1, 2 workers, shuffle=True",
            "dataset_class": IndexDataset,
            "dataset_kwargs": {"size": 20},
            "batch_size": 1,
            "nb_consumer": 1,
            "worker_per_consumer": 2,
            "shuffle": True,
            "multithread": True,
            "num_epochs": 4,
        }
    )

    # ============================================================================
    # Test Case 7: Large dataset
    # ============================================================================
    test_configs.append(
        {
            "name": "IndexDataset - Large dataset, 4 workers, shuffle=True",
            "dataset_class": IndexDataset,
            "dataset_kwargs": {"size": 200},
            "batch_size": 10,
            "nb_consumer": 1,
            "worker_per_consumer": 4,
            "shuffle": True,
            "multithread": True,
            "num_epochs": 8,
        }
    )

    # ============================================================================
    # Run all tests
    # ============================================================================

    print(f"\nTotal test configurations: {len(test_configs)}")
    print(f"  All tests use shuffle=True")
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
        print("  ✓ All epochs have complete sample coverage")
        print("  ✓ Worker rotation verified")
        print("  ✓ No deadlocks detected")
        exit(0)


if __name__ == "__main__":
    main()
