"""
Debug test to understand missing samples issue with shuffle=True

This test focuses on the specific failing case to understand why samples are missing.
"""

import numpy as np
from swift_loader import SwiftLoader
import time


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


def test_debug_missing_samples():
    """Debug test for missing samples with shuffle=True"""
    dataset_size = 25
    batch_size = 5
    num_workers = 4

    print("=" * 80)
    print("DEBUG TEST: Missing Samples with shuffle=True")
    print("=" * 80)
    print(f"Dataset size: {dataset_size}")
    print(f"Batch size: {batch_size}")
    print(f"Number of workers: {num_workers}")
    print()

    # Calculate expected worker segments
    worker_size = int(np.ceil(dataset_size / num_workers))
    print(f"Worker size (ceil): {worker_size}")
    print()

    for w in range(num_workers):
        start = w * worker_size
        stop = min(start + worker_size, dataset_size)
        num_samples = stop - start
        num_batches = int(np.ceil(num_samples / batch_size))
        expected_indices = list(range(start, stop))
        print(f"Worker {w}: samples {start}-{stop-1} ({num_samples} samples, {num_batches} batches)")
        print(f"  Expected indices: {expected_indices}")
    print()

    # Create loader with detailed logging
    loader = SwiftLoader(
        dataset_class=IndexDataset,
        dataset_kwargs={"size": dataset_size},
        batch_size=batch_size,
        nb_consumer=1,
        worker_per_consumer=num_workers,
        shuffle=True,
        multithread=True,
        data_queue_size=10,
        logger={"path": "/tmp/swift_loader_debug.log", "stdout": True, "level": "DEBUG"},
    )

    loader.start(consumer_index=0)

    print(f"Expected total batches: {len(loader)}")
    print()

    # Collect samples from first epoch with detailed tracking
    all_samples = []
    batch_count = 0
    epoch_samples = []

    print("Collecting batches from epoch 1...")
    print("-" * 80)

    try:
        for batch in loader:
            batch_count += 1

            # Convert batch to list
            if hasattr(batch, "tolist"):
                batch_list = batch.tolist()
            elif isinstance(batch, (list, tuple)):
                batch_list = list(batch)
            else:
                batch_list = [int(x) for x in batch]

            # Flatten if nested
            if batch_list and isinstance(batch_list[0], (list, tuple)):
                batch_list = [item for sublist in batch_list for item in sublist]

            epoch_samples.extend(batch_list)
            all_samples.extend(batch_list)

            print(f"Batch {batch_count}: {batch_list} (total samples so far: {len(epoch_samples)})")

            # Safety check - don't run forever
            if batch_count > 20:
                print("WARNING: Too many batches, stopping")
                break

    except StopIteration:
        print("StopIteration raised")
    except Exception as e:
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()

    print("-" * 80)
    print(f"\nEpoch 1 Summary:")
    print(f"  Total batches collected: {batch_count}")
    print(f"  Total samples collected: {len(epoch_samples)}")
    print(f"  Expected samples: {dataset_size}")
    print()

    # Analyze what we got
    unique_samples = set(epoch_samples)
    expected_samples = set(range(dataset_size))

    print(f"Unique samples collected: {sorted(unique_samples)}")
    print(f"Expected samples: {sorted(expected_samples)}")
    print()

    missing = expected_samples - unique_samples
    extra = unique_samples - expected_samples
    duplicates = len(epoch_samples) - len(unique_samples)

    if missing:
        print(f"❌ MISSING SAMPLES: {sorted(missing)}")
    if extra:
        print(f"❌ EXTRA SAMPLES: {sorted(extra)}")
    if duplicates:
        print(f"❌ DUPLICATES: {duplicates} duplicate samples found")
        # Find which samples are duplicated
        from collections import Counter
        counts = Counter(epoch_samples)
        duplicates_list = [(s, c) for s, c in counts.items() if c > 1]
        print(f"   Duplicated samples: {duplicates_list}")

    if not missing and not extra and not duplicates:
        print("✓ All samples present, no duplicates, no extras!")

    print()
    print("Sample order in epoch:")
    print(f"  {epoch_samples}")

    loader.close()


if __name__ == "__main__":
    test_debug_missing_samples()

