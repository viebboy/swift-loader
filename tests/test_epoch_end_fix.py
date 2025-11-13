#!/usr/bin/env python
"""
Quick test to verify epoch end signal fix
Run with: python tests/test_epoch_end_fix.py
"""

import time
from swift_loader import SwiftLoader


class SimpleDataset:
    def __init__(self, size=100):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return idx


print("Testing shuffle=False with epoch end fix...")
print("=" * 60)

loader = SwiftLoader(
    dataset_class=SimpleDataset,
    dataset_kwargs={"size": 100},
    batch_size=10,
    nb_consumer=1,
    worker_per_consumer=2,
    shuffle=False,
    seed=42,
    data_queue_size=5,
)

print("Starting loader...")
start = time.time()
loader.start(consumer_index=0)
print(f"Started in {time.time() - start:.2f}s")

try:
    print("\nEpoch 1:")
    batch_count = 0
    epoch_start = time.time()
    
    for batch in loader:
        batch_count += 1
        print(f"  Got batch {batch_count}")
        
        if batch_count >= 12:  # Should get ~10 batches, but limit to 12
            print(f"  Reached limit, stopping")
            break
    
    epoch_time = time.time() - epoch_start
    print(f"Epoch 1 completed: {batch_count} batches in {epoch_time:.2f}s")
    
    if epoch_time > 5:
        print(f"  WARNING: Still slow ({epoch_time:.2f}s)")
    else:
        print(f"  ✓ Fast enough ({epoch_time:.2f}s)")
    
except KeyboardInterrupt:
    print("\nInterrupted")
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("\nClosing...")
    loader.close()
    print("Done")

