"""
Test script for swift_loader with numpy array datasets

This script tests various configurations with datasets that return random numpy arrays.
It creates and destroys data loaders for each configuration and shows progress.

Usage:
    python tests/test_script_numpy.py

The script will:
    - Test 14+ different configurations
    - Create and destroy data loaders for each configuration
    - Show progress bars for each epoch
    - Print summary statistics

Optional dependencies:
    - tqdm: For better progress bars (falls back to simple progress if not available)
      Install with: pip install tqdm
"""

import numpy as np
from swift_loader import SwiftLoader
import time
import sys
import signal
from contextlib import contextmanager

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


class RandomNumpyTupleDataset:
    """Dataset that returns tuples of (array, label)"""

    def __init__(self, size: int, array_shape: tuple = (28, 28), num_classes: int = 10):
        self.size = size
        self.array_shape = array_shape
        self.num_classes = num_classes
        np.random.seed(42)
        self.data = [
            (
                np.random.rand(*array_shape).astype(np.float32),
                np.random.randint(0, num_classes),
            )
            for _ in range(size)
        ]

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int) -> tuple:
        if idx >= self.size or idx < 0:
            raise IndexError(
                f"Index {idx} out of range for dataset of size {self.size}"
            )
        return self.data[idx]


class RandomNumpyNestedDataset:
    """Dataset that returns nested structures (list of arrays)"""

    def __init__(self, size: int, num_arrays: int = 3, array_shape: tuple = (10,)):
        self.size = size
        self.num_arrays = num_arrays
        self.array_shape = array_shape
        np.random.seed(42)
        self.data = [
            [np.random.rand(*array_shape).astype(np.float32) for _ in range(num_arrays)]
            for _ in range(size)
        ]

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int) -> list:
        if idx >= self.size or idx < 0:
            raise IndexError(
                f"Index {idx} out of range for dataset of size {self.size}"
            )
        return self.data[idx]


def test_configuration(
    config_name: str,
    dataset_class,
    dataset_kwargs: dict,
    batch_size: int,
    nb_consumer: int,
    worker_per_consumer: int,
    shuffle: bool,
    num_epochs: int = 5,
    multithread: bool = True,
    data_queue_size: int = 10,
    timeout_seconds: int = 60,
    **kwargs,
):
    """Test a specific configuration"""
    print(f"\n{'='*80}")
    print(f"Testing: {config_name}")
    print(f"{'='*80}")
    print(f"  Dataset: {dataset_class.__name__}")
    print(f"  Dataset size: {dataset_kwargs.get('size', 'N/A')}")
    print(f"  Batch size: {batch_size}")
    print(f"  Consumers: {nb_consumer}, Workers per consumer: {worker_per_consumer}")
    print(f"  Shuffle: {shuffle}")
    print(f"  Multithread: {multithread}")
    print(f"  Epochs: {num_epochs}")
    print(f"{'='*80}\n")

    loader = None
    try:
        # Create loader
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

        # Start loader
        loader.start(consumer_index=0)

        # Get expected number of batches
        expected_batches = len(loader)
        print(f"Expected batches per epoch: {expected_batches}")

        # Test multiple epochs with timeout
        total_batches = 0
        try:
            with timeout_context(timeout_seconds * num_epochs):
                for epoch in range(num_epochs):
                    epoch_batches = 0
                    epoch_samples = 0

                    # Create progress bar for this epoch
                    pbar = tqdm(
                        desc=f"Epoch {epoch+1}/{num_epochs}",
                        total=expected_batches,
                        unit="batch",
                        ncols=100,
                    )

                    try:
                        for batch in loader:
                            epoch_batches += 1
                            epoch_samples += (
                                len(batch) if hasattr(batch, "__len__") else 1
                            )
                            total_batches += 1
                            pbar.update(1)

                            # Verify batch structure
                            if isinstance(batch, (list, tuple)):
                                # Handle tuple/list returns
                                for item in batch:
                                    if isinstance(item, np.ndarray):
                                        assert item.dtype in [
                                            np.float32,
                                            np.float64,
                                            np.int32,
                                            np.int64,
                                        ], f"Unexpected dtype: {item.dtype}"

                    except StopIteration:
                        pass
                    finally:
                        pbar.close()

                    print(
                        f"  Epoch {epoch+1}: {epoch_batches} batches, ~{epoch_samples} samples"
                    )
        except TimeoutError as e:
            print(f"\n⚠️  TIMEOUT: {e}")
            print("  This configuration appears to be stuck (likely a deadlock).")
            print(f"  Configuration: {config_name}")
            if not shuffle:
                print(
                    "  Note: shuffle=False is known to have deadlock issues "
                    "with OrderDataQueue"
                )
            raise

        print(f"\n✓ Configuration '{config_name}' completed successfully!")
        print(f"  Total batches processed: {total_batches}")
        print(f"  Average batches per epoch: {total_batches / num_epochs:.2f}")

    except Exception as e:
        print(f"\n✗ Configuration '{config_name}' FAILED with error:")
        print(f"  {type(e).__name__}: {str(e)}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        # Clean up
        if loader is not None:
            try:
                loader.close()
            except Exception as e:
                print(f"  Warning: Error during cleanup: {e}")


def main():
    """Run all test configurations"""

    print("\n" + "=" * 80)
    print("Swift Loader Comprehensive Test Script - Numpy Arrays")
    print("=" * 80)

    test_configs = []

    # ============================================================================
    # Test Case 1: Simple numpy arrays - Single consumer, single worker
    # ============================================================================
    test_configs.append(
        {
            "name": "Simple arrays - 1 consumer, 1 worker, shuffle=True, multithread=True",
            "dataset_class": RandomNumpyDataset,
            "dataset_kwargs": {"size": 50, "shape": (32, 32, 3)},
            "batch_size": 5,
            "nb_consumer": 1,
            "worker_per_consumer": 1,
            "shuffle": True,
            "multithread": True,
            "num_epochs": 3,
        }
    )

    test_configs.append(
        {
            "name": "Simple arrays - 1 consumer, 1 worker, shuffle=False, multithread=True",
            "dataset_class": RandomNumpyDataset,
            "dataset_kwargs": {"size": 50, "shape": (32, 32, 3)},
            "batch_size": 5,
            "nb_consumer": 1,
            "worker_per_consumer": 1,
            "shuffle": False,
            "multithread": True,
            "num_epochs": 3,
        }
    )

    # ============================================================================
    # Test Case 2: Simple numpy arrays - Single consumer, multiple workers
    # ============================================================================
    test_configs.append(
        {
            "name": "Simple arrays - 1 consumer, 3 workers, shuffle=True",
            "dataset_class": RandomNumpyDataset,
            "dataset_kwargs": {"size": 60, "shape": (28, 28)},
            "batch_size": 4,
            "nb_consumer": 1,
            "worker_per_consumer": 3,
            "shuffle": True,
            "num_epochs": 4,
        }
    )

    test_configs.append(
        {
            "name": "Simple arrays - 1 consumer, 3 workers, shuffle=False",
            "dataset_class": RandomNumpyDataset,
            "dataset_kwargs": {"size": 60, "shape": (28, 28)},
            "batch_size": 4,
            "nb_consumer": 1,
            "worker_per_consumer": 3,
            "shuffle": False,
            "num_epochs": 4,
        }
    )

    # ============================================================================
    # Test Case 3: Tuple dataset (array, label)
    # ============================================================================
    test_configs.append(
        {
            "name": "Tuple dataset (array, label) - 1 consumer, 2 workers",
            "dataset_class": RandomNumpyTupleDataset,
            "dataset_kwargs": {"size": 40, "array_shape": (32, 32), "num_classes": 10},
            "batch_size": 5,
            "nb_consumer": 1,
            "worker_per_consumer": 2,
            "shuffle": True,
            "num_epochs": 3,
        }
    )

    # ============================================================================
    # Test Case 4: Nested dataset (list of arrays)
    # ============================================================================
    test_configs.append(
        {
            "name": "Nested dataset (list of arrays) - 1 consumer, 2 workers",
            "dataset_class": RandomNumpyNestedDataset,
            "dataset_kwargs": {"size": 30, "num_arrays": 3, "array_shape": (10,)},
            "batch_size": 4,
            "nb_consumer": 1,
            "worker_per_consumer": 2,
            "shuffle": True,
            "num_epochs": 3,
        }
    )

    # ============================================================================
    # Test Case 5: Large dataset with uneven division
    # ============================================================================
    test_configs.append(
        {
            "name": "Large dataset (prime size) - 1 consumer, 4 workers",
            "dataset_class": RandomNumpyDataset,
            "dataset_kwargs": {"size": 97, "shape": (16, 16)},  # Prime number
            "batch_size": 7,
            "nb_consumer": 1,
            "worker_per_consumer": 4,
            "shuffle": True,
            "num_epochs": 5,
        }
    )

    test_configs.append(
        {
            "name": "Large dataset (prime size) - 1 consumer, 4 workers, shuffle=False",
            "dataset_class": RandomNumpyDataset,
            "dataset_kwargs": {"size": 97, "shape": (16, 16)},
            "batch_size": 7,
            "nb_consumer": 1,
            "worker_per_consumer": 4,
            "shuffle": False,
            "num_epochs": 5,
        }
    )

    # ============================================================================
    # Test Case 6: Small batch size
    # ============================================================================
    test_configs.append(
        {
            "name": "Small batch size - 1 consumer, 3 workers",
            "dataset_class": RandomNumpyDataset,
            "dataset_kwargs": {"size": 30, "shape": (10, 10)},
            "batch_size": 1,
            "nb_consumer": 1,
            "worker_per_consumer": 3,
            "shuffle": True,
            "num_epochs": 3,
        }
    )

    # ============================================================================
    # Test Case 7: Large batch size
    # ============================================================================
    test_configs.append(
        {
            "name": "Large batch size - 1 consumer, 2 workers",
            "dataset_class": RandomNumpyDataset,
            "dataset_kwargs": {"size": 20, "shape": (8, 8)},
            "batch_size": 10,
            "nb_consumer": 1,
            "worker_per_consumer": 2,
            "shuffle": True,
            "num_epochs": 3,
        }
    )

    # ============================================================================
    # Test Case 8: Different array shapes
    # ============================================================================
    test_configs.append(
        {
            "name": "1D arrays - 1 consumer, 2 workers",
            "dataset_class": RandomNumpyDataset,
            "dataset_kwargs": {"size": 40, "shape": (100,)},
            "batch_size": 5,
            "nb_consumer": 1,
            "worker_per_consumer": 2,
            "shuffle": True,
            "num_epochs": 3,
        }
    )

    test_configs.append(
        {
            "name": "3D arrays - 1 consumer, 2 workers",
            "dataset_class": RandomNumpyDataset,
            "dataset_kwargs": {"size": 30, "shape": (10, 10, 3)},
            "batch_size": 4,
            "nb_consumer": 1,
            "worker_per_consumer": 2,
            "shuffle": True,
            "num_epochs": 3,
        }
    )

    # ============================================================================
    # Test Case 9: Explicit Multithread=True tests
    # ============================================================================
    test_configs.append(
        {
            "name": "Multithread=True - 1 consumer, 2 workers, shuffle=True",
            "dataset_class": RandomNumpyDataset,
            "dataset_kwargs": {"size": 50, "shape": (20, 20)},
            "batch_size": 5,
            "nb_consumer": 1,
            "worker_per_consumer": 2,
            "shuffle": True,
            "multithread": True,
            "num_epochs": 3,
        }
    )

    test_configs.append(
        {
            "name": "Multithread=True - 1 consumer, 2 workers, shuffle=False",
            "dataset_class": RandomNumpyDataset,
            "dataset_kwargs": {"size": 50, "shape": (20, 20)},
            "batch_size": 5,
            "nb_consumer": 1,
            "worker_per_consumer": 2,
            "shuffle": False,
            "multithread": True,
            "num_epochs": 3,
        }
    )

    test_configs.append(
        {
            "name": "Multithread=True - 1 consumer, 4 workers, shuffle=True",
            "dataset_class": RandomNumpyDataset,
            "dataset_kwargs": {"size": 80, "shape": (15, 15)},
            "batch_size": 4,
            "nb_consumer": 1,
            "worker_per_consumer": 4,
            "shuffle": True,
            "multithread": True,
            "num_epochs": 4,
        }
    )

    test_configs.append(
        {
            "name": "Multithread=True - 1 consumer, 4 workers, shuffle=False",
            "dataset_class": RandomNumpyDataset,
            "dataset_kwargs": {"size": 80, "shape": (15, 15)},
            "batch_size": 4,
            "nb_consumer": 1,
            "worker_per_consumer": 4,
            "shuffle": False,
            "multithread": True,
            "num_epochs": 4,
        }
    )

    test_configs.append(
        {
            "name": "Multithread=True - Uneven dataset, 3 workers, shuffle=True",
            "dataset_class": RandomNumpyDataset,
            "dataset_kwargs": {"size": 47, "shape": (12, 12)},  # Prime number
            "batch_size": 5,
            "nb_consumer": 1,
            "worker_per_consumer": 3,
            "shuffle": True,
            "multithread": True,
            "num_epochs": 5,
        }
    )

    test_configs.append(
        {
            "name": "Multithread=True - Uneven dataset, 3 workers, shuffle=False",
            "dataset_class": RandomNumpyDataset,
            "dataset_kwargs": {"size": 47, "shape": (12, 12)},
            "batch_size": 5,
            "nb_consumer": 1,
            "worker_per_consumer": 3,
            "shuffle": False,
            "multithread": True,
            "num_epochs": 5,
        }
    )

    test_configs.append(
        {
            "name": "Multithread=True - Tuple dataset, 3 workers",
            "dataset_class": RandomNumpyTupleDataset,
            "dataset_kwargs": {"size": 45, "array_shape": (20, 20), "num_classes": 5},
            "batch_size": 5,
            "nb_consumer": 1,
            "worker_per_consumer": 3,
            "shuffle": True,
            "multithread": True,
            "num_epochs": 3,
        }
    )

    test_configs.append(
        {
            "name": "Multithread=True - Nested dataset, 2 workers",
            "dataset_class": RandomNumpyNestedDataset,
            "dataset_kwargs": {"size": 40, "num_arrays": 4, "array_shape": (8,)},
            "batch_size": 4,
            "nb_consumer": 1,
            "worker_per_consumer": 2,
            "shuffle": True,
            "multithread": True,
            "num_epochs": 3,
        }
    )

    test_configs.append(
        {
            "name": "Multithread=True - Large queue, 3 workers",
            "dataset_class": RandomNumpyDataset,
            "dataset_kwargs": {"size": 60, "shape": (10, 10)},
            "batch_size": 5,
            "nb_consumer": 1,
            "worker_per_consumer": 3,
            "shuffle": True,
            "multithread": True,
            "data_queue_size": 20,
            "num_epochs": 3,
        }
    )

    test_configs.append(
        {
            "name": "Multithread=True - Rotation test, 4 workers, many epochs",
            "dataset_class": RandomNumpyDataset,
            "dataset_kwargs": {"size": 50, "shape": (12, 12)},
            "batch_size": 4,
            "nb_consumer": 1,
            "worker_per_consumer": 4,
            "shuffle": True,
            "multithread": True,
            "num_epochs": 8,  # Enough to see rotation
        }
    )

    # ============================================================================
    # Test Case 10: Multithread=False (for comparison)
    # ============================================================================
    test_configs.append(
        {
            "name": "Multithread=False - 1 consumer, 3 workers, shuffle=True",
            "dataset_class": RandomNumpyDataset,
            "dataset_kwargs": {"size": 50, "shape": (20, 20)},
            "batch_size": 5,
            "nb_consumer": 1,
            "worker_per_consumer": 3,
            "shuffle": True,
            "multithread": False,
            "num_epochs": 3,
        }
    )

    test_configs.append(
        {
            "name": "Multithread=False - 1 consumer, 3 workers, shuffle=False",
            "dataset_class": RandomNumpyDataset,
            "dataset_kwargs": {"size": 50, "shape": (20, 20)},
            "batch_size": 5,
            "nb_consumer": 1,
            "worker_per_consumer": 3,
            "shuffle": False,
            "multithread": False,
            "num_epochs": 3,
        }
    )

    # ============================================================================
    # Test Case 11: Very small dataset
    # ============================================================================
    test_configs.append(
        {
            "name": "Very small dataset - 1 consumer, 2 workers",
            "dataset_class": RandomNumpyDataset,
            "dataset_kwargs": {"size": 5, "shape": (5, 5)},
            "batch_size": 2,
            "nb_consumer": 1,
            "worker_per_consumer": 2,
            "shuffle": True,
            "num_epochs": 3,
        }
    )

    # ============================================================================
    # Test Case 12: Rotation test (many epochs) - already covered above
    # ============================================================================
    test_configs.append(
        {
            "name": "Rotation test (many epochs) - 1 consumer, 4 workers",
            "dataset_class": RandomNumpyDataset,
            "dataset_kwargs": {"size": 40, "shape": (15, 15)},
            "batch_size": 4,
            "nb_consumer": 1,
            "worker_per_consumer": 4,
            "shuffle": True,
            "num_epochs": 8,  # Enough to see rotation
        }
    )

    # ============================================================================
    # Test Case 13: Different dtypes
    # ============================================================================
    test_configs.append(
        {
            "name": "Float64 arrays - 1 consumer, 2 workers",
            "dataset_class": RandomNumpyDataset,
            "dataset_kwargs": {"size": 30, "shape": (12, 12), "dtype": np.float64},
            "batch_size": 4,
            "nb_consumer": 1,
            "worker_per_consumer": 2,
            "shuffle": True,
            "num_epochs": 3,
        }
    )

    # ============================================================================
    # Test Case 14: Large queue size - already covered above
    # ============================================================================
    test_configs.append(
        {
            "name": "Large queue size - 1 consumer, 2 workers",
            "dataset_class": RandomNumpyDataset,
            "dataset_kwargs": {"size": 50, "shape": (10, 10)},
            "batch_size": 5,
            "nb_consumer": 1,
            "worker_per_consumer": 2,
            "shuffle": True,
            "data_queue_size": 20,
            "num_epochs": 3,
        }
    )

    # ============================================================================
    # Test Case 15: Multiple consumers (simulated by testing consumer 0)
    # ============================================================================
    test_configs.append(
        {
            "name": "Multiple consumers setup - consumer 0, 2 workers",
            "dataset_class": RandomNumpyDataset,
            "dataset_kwargs": {"size": 60, "shape": (14, 14)},
            "batch_size": 5,
            "nb_consumer": 3,  # 3 consumers total
            "worker_per_consumer": 2,
            "shuffle": True,
            "num_epochs": 3,
        }
    )

    # ============================================================================
    # Run all tests
    # ============================================================================

    print(f"\nTotal test configurations: {len(test_configs)}")
    print(
        f"  - Multithread=True tests: {sum(1 for c in test_configs if c.get('multithread', True) == True)}"
    )
    print(
        f"  - Multithread=False tests: {sum(1 for c in test_configs if c.get('multithread', True) == False)}"
    )

    print()

    start_time = time.time()
    passed = 0
    failed = 0

    for i, config in enumerate(test_configs, 1):
        print(f"\n[{i}/{len(test_configs)}]")

        try:
            # Extract 'name' and pass it as 'config_name'
            # Make a copy to avoid modifying the original
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
        exit(0)


if __name__ == "__main__":
    main()
