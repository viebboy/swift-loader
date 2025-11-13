#!/usr/bin/env python
"""
Detailed debug script to see what's happening
Run with: python tests/test_debug_detailed.py
"""

import time
import sys
import threading
from swift_loader import SwiftLoader


class SimpleDataset:
    def __init__(self, size=50):  # Smaller for faster testing
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return idx


print("=" * 60)
print("Detailed Debug Test - shuffle=False")
print("=" * 60)

loader = SwiftLoader(
    dataset_class=SimpleDataset,
    dataset_kwargs={"size": 50},
    batch_size=5,
    nb_consumer=1,
    worker_per_consumer=2,
    shuffle=False,
    seed=42,
    data_queue_size=3,  # Very small queue
)

print("\nStarting loader...")
start = time.time()
loader.start(consumer_index=0)
print(f"Started in {time.time() - start:.2f}s")

print("\nStarting to consume batches...")
print("(Press Ctrl+C to interrupt if it hangs)")

try:
    batch_count = 0
    last_print_time = time.time()
    
    for batch in loader:
        batch_count += 1
        current_time = time.time()
        elapsed = current_time - last_print_time
        
        print(f"Batch {batch_count}: got {len(batch) if hasattr(batch, '__len__') else '?'} items (waited {elapsed:.3f}s)")
        last_print_time = current_time
        
        if batch_count >= 15:
            print(f"\nReached limit of 15 batches")
            break
        
        # If we wait more than 2 seconds between batches, something is wrong
        if elapsed > 2.0:
            print(f"  WARNING: Long wait ({elapsed:.2f}s) between batches!")
    
    total_time = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"Completed: {batch_count} batches in {total_time:.2f}s")
    print(f"{'=' * 60}")
    
except KeyboardInterrupt:
    print("\n\nInterrupted by user")
    print(f"Got {batch_count} batches before interruption")
except Exception as e:
    print(f"\n\nERROR: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("\nClosing loader...")
    loader.close()
    print("Done")

