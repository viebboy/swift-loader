#!/usr/bin/env python
"""
Debug script to test shuffle=True performance (for comparison)
Run with: python tests/test_shuffle_true_debug.py
"""

import time
import sys
from swift_loader import SwiftLoader


class SimpleDataset:
    """Simple dataset that returns index as data"""
    
    def __init__(self, size=100):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return idx


def test_shuffle_true():
    """Test shuffle=True with timing and progress prints"""
    print("=" * 60)
    print("Testing shuffle=True with multiple workers (for comparison)")
    print("=" * 60)
    
    dataset_size = 100
    batch_size = 10
    nb_consumer = 1
    worker_per_consumer = 2
    
    print(f"\nConfiguration:")
    print(f"  Dataset size: {dataset_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Consumers: {nb_consumer}")
    print(f"  Workers per consumer: {worker_per_consumer}")
    print(f"  Shuffle: True")
    
    loader = SwiftLoader(
        dataset_class=SimpleDataset,
        dataset_kwargs={"size": dataset_size},
        batch_size=batch_size,
        nb_consumer=nb_consumer,
        worker_per_consumer=worker_per_consumer,
        shuffle=True,
        seed=42,
        data_queue_size=5,
    )
    
    print("\nStarting loader...")
    start_time = time.time()
    loader.start(consumer_index=0)
    init_time = time.time() - start_time
    print(f"Loader started in {init_time:.2f}s")
    
    try:
        num_epochs = 2
        print(f"\nProcessing {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch + 1} ---")
            epoch_start = time.time()
            batch_count = 0
            sample_count = 0
            max_batches = 15
            
            for batch in loader:
                batch_count += 1
                if isinstance(batch, (list, tuple)):
                    sample_count += len(batch)
                else:
                    try:
                        sample_count += len(batch) if hasattr(batch, '__len__') else 1
                    except:
                        sample_count += 1
                
                if batch_count % 5 == 0:
                    elapsed = time.time() - epoch_start
                    print(f"  Batch {batch_count}: {sample_count} samples, {elapsed:.2f}s elapsed")
                
                if batch_count >= max_batches:
                    print(f"  Reached batch limit ({max_batches}), stopping epoch")
                    break
            
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch + 1} completed: {batch_count} batches, {sample_count} samples, {epoch_time:.2f}s")
        
        total_time = time.time() - start_time
        print(f"\n{'=' * 60}")
        print(f"Total time: {total_time:.2f}s")
        print(f"{'=' * 60}")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nClosing loader...")
        loader.close()
        print("Done")


if __name__ == "__main__":
    test_shuffle_true()

