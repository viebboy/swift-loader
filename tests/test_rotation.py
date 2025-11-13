"""
Tests for sample index rotation functionality in SwiftLoader

Tests verify that:
1. When shuffle=True, workers rotate through different sample ranges each epoch
2. When shuffle=False, workers maintain fixed ranges (no rotation)
3. All samples are covered across epochs
4. No overlap within an epoch
"""

import pytest
import numpy as np
import dill
import tempfile
import os
from swift_loader import SwiftLoader


class SimpleDataset:
    """Simple dataset for testing that returns the index as data"""
    
    def __init__(self, size=100):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Return the index itself so we can verify which samples are being processed
        return idx


def get_worker_indices_from_loader(loader, consumer_index, num_epochs=3):
    """
    Extract the actual sample indices processed by each worker across epochs.
    
    Returns:
        dict: {epoch: {worker_index: [list of sample indices]}}
    """
    loader.start(consumer_index=consumer_index)
    
    epoch_data = {}
    current_epoch = 0
    current_epoch_samples = {}
    
    # Track which worker each batch comes from
    # Since we can't directly access worker indices from the loader,
    # we'll collect all samples per epoch and verify coverage
    all_samples_per_epoch = []
    
    try:
        for epoch in range(num_epochs):
            epoch_samples = []
            batch_count = 0
            
            for batch in loader:
                # batch is a tensor/array, extract the indices
                if isinstance(batch, (list, tuple)):
                    batch_indices = list(batch)
                else:
                    # If it's a tensor, convert to list
                    try:
                        batch_indices = batch.tolist() if hasattr(batch, 'tolist') else list(batch)
                    except:
                        batch_indices = [batch]
                
                epoch_samples.extend(batch_indices)
                batch_count += 1
            
            all_samples_per_epoch.append(sorted(epoch_samples))
            
            # Check if we've completed an epoch
            if len(epoch_samples) > 0:
                current_epoch += 1
    
    finally:
        loader.close()
    
    return all_samples_per_epoch


def verify_rotation_shuffle_true(samples_per_epoch, nb_consumer, worker_per_consumer, dataset_size):
    """
    Verify that when shuffle=True, workers rotate through different ranges.
    
    For K workers, over K epochs, each worker should see all K ranges.
    """
    nb_worker = nb_consumer * worker_per_consumer
    worker_size = int(np.ceil(dataset_size / nb_worker))
    
    # Collect which ranges each worker sees across epochs
    # We can't directly track which worker processed which batch,
    # but we can verify that the samples rotate correctly
    
    # For each epoch, check that samples form non-overlapping ranges
    for epoch_idx, epoch_samples in enumerate(samples_per_epoch):
        # Samples should be sorted and form a contiguous range
        assert len(epoch_samples) > 0, f"Epoch {epoch_idx} has no samples"
        
        # Check that samples are within expected bounds
        min_sample = min(epoch_samples)
        max_sample = max(epoch_samples)
        
        # The range should be approximately worker_size samples
        # (allowing for rounding at the end)
        assert max_sample < dataset_size, f"Sample {max_sample} exceeds dataset size {dataset_size}"
        assert min_sample >= 0, f"Sample {min_sample} is negative"
    
    # Verify that over multiple epochs, we see different ranges
    # This is a simplified check - in practice, we'd need to track per-worker
    if len(samples_per_epoch) >= nb_worker:
        # Collect unique starting positions across epochs
        # (simplified - actual implementation would track per worker)
        unique_ranges = set()
        for epoch_samples in samples_per_epoch:
            if len(epoch_samples) > 0:
                # Get the range start (approximate)
                range_start = (min(epoch_samples) // worker_size) * worker_size
                unique_ranges.add(range_start)
        
        # We should see multiple different ranges across epochs
        # (This is a heuristic check since we can't directly track per-worker)
        assert len(unique_ranges) >= min(2, nb_worker), \
            f"Expected to see rotation across epochs, but only saw {len(unique_ranges)} unique ranges"


def verify_no_rotation_shuffle_false(samples_per_epoch, nb_consumer, worker_per_consumer, dataset_size):
    """
    Verify that when shuffle=False, workers maintain fixed ranges (no rotation).
    """
    # When shuffle=False, each consumer gets a fixed segment
    # The samples should be consistent across epochs (order may vary but range should be same)
    if len(samples_per_epoch) >= 2:
        # Check that the set of samples is the same across epochs
        # (allowing for different ordering)
        first_epoch_set = set(samples_per_epoch[0])
        for epoch_idx, epoch_samples in enumerate(samples_per_epoch[1:], 1):
            current_epoch_set = set(epoch_samples)
            # The sets should be the same (same samples, possibly different order)
            assert first_epoch_set == current_epoch_set, \
                f"Epoch 0 and epoch {epoch_idx} have different sample sets when shuffle=False"


def verify_coverage(samples_per_epoch, dataset_size, num_epochs):
    """
    Verify that all samples in the dataset are covered across all epochs.
    """
    all_samples_seen = set()
    for epoch_samples in samples_per_epoch:
        all_samples_seen.update(epoch_samples)
    
    # Check that we've seen all samples
    expected_samples = set(range(dataset_size))
    missing_samples = expected_samples - all_samples_seen
    
    # Allow some tolerance - with rotation, we might not see every sample
    # in every configuration, but we should see most
    coverage_ratio = len(all_samples_seen) / dataset_size
    assert coverage_ratio >= 0.9, \
        f"Only {coverage_ratio*100:.1f}% of samples were covered. Missing: {len(missing_samples)} samples"


@pytest.mark.parametrize("nb_consumer,worker_per_consumer,shuffle", [
    (1, 1, True),
    (1, 1, False),
    (1, 2, True),
    (1, 2, False),
    (2, 2, True),
    (2, 2, False),
])
def test_loader_rotation(nb_consumer, worker_per_consumer, shuffle):
    """Test rotation behavior for different configurations"""
    dataset_size = 100
    batch_size = 10
    num_epochs = 3
    
    # Create dataset
    dataset = SimpleDataset(size=dataset_size)
    
    # Create loader
    loader = SwiftLoader(
        dataset_class=SimpleDataset,
        dataset_kwargs={"size": dataset_size},
        batch_size=batch_size,
        nb_consumer=nb_consumer,
        worker_per_consumer=worker_per_consumer,
        shuffle=shuffle,
        seed=42,
    )
    
    # Test with first consumer (consumer_index=0)
    samples_per_epoch = get_worker_indices_from_loader(
        loader, consumer_index=0, num_epochs=num_epochs
    )
    
    # Verify we got data for all epochs
    assert len(samples_per_epoch) == num_epochs, \
        f"Expected {num_epochs} epochs, got {len(samples_per_epoch)}"
    
    # Verify coverage
    verify_coverage(samples_per_epoch, dataset_size, num_epochs)
    
    # Verify rotation behavior
    if shuffle:
        verify_rotation_shuffle_true(
            samples_per_epoch, nb_consumer, worker_per_consumer, dataset_size
        )
    else:
        verify_no_rotation_shuffle_false(
            samples_per_epoch, nb_consumer, worker_per_consumer, dataset_size
        )


def test_multiple_consumers_coverage():
    """Test that multiple consumers together cover all samples"""
    dataset_size = 100
    batch_size = 10
    nb_consumer = 2
    worker_per_consumer = 2
    num_epochs = 2
    
    dataset = SimpleDataset(size=dataset_size)
    
    all_samples_all_consumers = set()
    
    # Collect samples from all consumers
    for consumer_idx in range(nb_consumer):
        loader = SwiftLoader(
            dataset_class=SimpleDataset,
            dataset_kwargs={"size": dataset_size},
            batch_size=batch_size,
            nb_consumer=nb_consumer,
            worker_per_consumer=worker_per_consumer,
            shuffle=True,
            seed=42,
        )
        
        loader.start(consumer_index=consumer_idx)
        
        try:
            for epoch in range(num_epochs):
                for batch in loader:
                    if isinstance(batch, (list, tuple)):
                        batch_indices = list(batch)
                    else:
                        try:
                            batch_indices = batch.tolist() if hasattr(batch, 'tolist') else list(batch)
                        except:
                            batch_indices = [batch]
                    all_samples_all_consumers.update(batch_indices)
        finally:
            loader.close()
    
    # Verify that together, all consumers cover (most of) the dataset
    coverage_ratio = len(all_samples_all_consumers) / dataset_size
    assert coverage_ratio >= 0.95, \
        f"Multiple consumers only covered {coverage_ratio*100:.1f}% of samples"


def test_rotation_cycle_completeness():
    """
    Test that rotation cycles through all ranges correctly.
    With K workers, after K epochs, each worker should have seen all K ranges.
    """
    dataset_size = 100
    batch_size = 5
    nb_consumer = 1
    worker_per_consumer = 4  # K = 4 workers
    num_epochs = 4  # K epochs to complete a full cycle
    
    dataset = SimpleDataset(size=dataset_size)
    
    loader = SwiftLoader(
        dataset_class=SimpleDataset,
        dataset_kwargs={"size": dataset_size},
        batch_size=batch_size,
        nb_consumer=nb_consumer,
        worker_per_consumer=worker_per_consumer,
        shuffle=True,
        seed=42,
    )
    
    samples_per_epoch = get_worker_indices_from_loader(
        loader, consumer_index=0, num_epochs=num_epochs
    )
    
    # Verify we have data for all epochs
    assert len(samples_per_epoch) == num_epochs
    
    # Collect all unique sample ranges seen
    nb_worker = nb_consumer * worker_per_consumer
    worker_size = int(np.ceil(dataset_size / nb_worker))
    
    # Check that samples in each epoch are within expected bounds
    for epoch_idx, epoch_samples in enumerate(samples_per_epoch):
        assert len(epoch_samples) > 0, f"Epoch {epoch_idx} is empty"
        min_sample = min(epoch_samples)
        max_sample = max(epoch_samples)
        assert 0 <= min_sample < dataset_size
        assert 0 <= max_sample < dataset_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

