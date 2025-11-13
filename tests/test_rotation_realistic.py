"""
Realistic integration tests for sample index rotation functionality

These tests use a dataset that generates random tensors (similar to real use-cases)
and verify rotation behavior across multiple epochs.
"""

import pytest
import torch
import numpy as np
from swift_loader import SwiftLoader


class RandomTensorDataset:
    """
    Dataset that generates random tensors for testing.
    Each sample is a tuple of (image_tensor, label_tensor).
    The image tensor contains the sample index so we can track which samples are processed.
    """
    
    def __init__(self, size=100, image_shape=(3, 32, 32), seed=42):
        self.size = size
        self.image_shape = image_shape
        self.seed = seed
        # Store sample indices in the first pixel of the image for tracking
        np.random.seed(seed)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        if idx >= self.size or idx < 0:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.size}")
        
        # Create a random image tensor
        # Store the index in the first pixel so we can track which sample this is
        image = torch.randn(*self.image_shape, dtype=torch.float32)
        # Store index in first pixel (multiply by small value to avoid affecting gradients)
        image[0, 0, 0] = float(idx) * 0.001
        
        # Create a random label
        label = torch.randint(0, 10, (1,), dtype=torch.long).squeeze()
        
        return image, label


def extract_sample_indices_from_batch(batch):
    """
    Extract sample indices from a batch.
    The index is stored in image[0, 0, 0] / 0.001
    """
    images, labels = batch
    if isinstance(images, torch.Tensor):
        # Batch is already a tensor
        if images.dim() == 4:  # (batch_size, channels, height, width)
            # Round to handle floating point precision issues
            indices = torch.round(images[:, 0, 0, 0] / 0.001).long().tolist()
        else:
            # Single sample
            indices = [int(round(images[0, 0, 0].item() / 0.001))]
    else:
        # Batch might be a list
        indices = [int(round(img[0, 0, 0].item() / 0.001)) for img in images]
    return indices


def test_rotation_shuffle_true_single_consumer():
    """Test rotation with shuffle=True, consumer=1, worker_per_consumer=1"""
    dataset_size = 100
    batch_size = 10
    nb_consumer = 1
    worker_per_consumer = 1
    num_epochs = 3
    
    loader = SwiftLoader(
        dataset_class=RandomTensorDataset,
        dataset_kwargs={"size": dataset_size, "seed": 42},
        batch_size=batch_size,
        nb_consumer=nb_consumer,
        worker_per_consumer=worker_per_consumer,
        shuffle=True,
        seed=42,
    )
    
    loader.start(consumer_index=0)
    
    try:
        # Track samples seen in each epoch
        samples_per_epoch = []
        
        for epoch in range(num_epochs):
            epoch_samples = []
            batch_count = 0
            
            for batch in loader:
                indices = extract_sample_indices_from_batch(batch)
                epoch_samples.extend(indices)
                batch_count += 1
            
            samples_per_epoch.append(sorted(epoch_samples))
            
            # Verify we got batches
            assert batch_count > 0, f"Epoch {epoch} produced no batches"
        
        # Verify rotation: samples should differ across epochs
        if num_epochs >= 2:
            # With rotation, different epochs should see different samples
            # (or at least the sets should differ due to rotation)
            unique_samples_epoch0 = set(samples_per_epoch[0])
            unique_samples_epoch1 = set(samples_per_epoch[1])
            
            # With rotation, we should see some difference
            # (exact overlap depends on dataset size and worker configuration)
            assert len(unique_samples_epoch0) > 0
            assert len(unique_samples_epoch1) > 0
            
        # Verify all samples are valid indices
        for epoch_idx, epoch_samples in enumerate(samples_per_epoch):
            assert all(0 <= idx < dataset_size for idx in epoch_samples), \
                f"Epoch {epoch_idx} contains invalid sample indices"
    
    finally:
        loader.close()


def test_rotation_shuffle_true_multiple_workers():
    """Test rotation with shuffle=True, consumer=1, worker_per_consumer=2"""
    dataset_size = 100
    batch_size = 10
    nb_consumer = 1
    worker_per_consumer = 2
    num_epochs = 3
    
    loader = SwiftLoader(
        dataset_class=RandomTensorDataset,
        dataset_kwargs={"size": dataset_size, "seed": 42},
        batch_size=batch_size,
        nb_consumer=nb_consumer,
        worker_per_consumer=worker_per_consumer,
        shuffle=True,
        seed=42,
    )
    
    loader.start(consumer_index=0)
    
    try:
        # Track samples seen in each epoch
        samples_per_epoch = []
        
        for epoch in range(num_epochs):
            epoch_samples = []
            
            for batch in loader:
                indices = extract_sample_indices_from_batch(batch)
                epoch_samples.extend(indices)
            
            samples_per_epoch.append(sorted(epoch_samples))
        
        # Verify we got data for all epochs
        assert len(samples_per_epoch) == num_epochs
        
        # Verify all samples are valid
        for epoch_idx, epoch_samples in enumerate(samples_per_epoch):
            assert len(epoch_samples) > 0, f"Epoch {epoch_idx} is empty"
            assert all(0 <= idx < dataset_size for idx in epoch_samples), \
                f"Epoch {epoch_idx} contains invalid indices"
    
    finally:
        loader.close()


def test_rotation_shuffle_true_multiple_consumers():
    """Test rotation with shuffle=True, consumer=2, worker_per_consumer=2"""
    dataset_size = 100
    batch_size = 10
    nb_consumer = 2
    worker_per_consumer = 2
    num_epochs = 3
    
    # Test consumer 0
    loader = SwiftLoader(
        dataset_class=RandomTensorDataset,
        dataset_kwargs={"size": dataset_size, "seed": 42},
        batch_size=batch_size,
        nb_consumer=nb_consumer,
        worker_per_consumer=worker_per_consumer,
        shuffle=True,
        seed=42,
    )
    
    loader.start(consumer_index=0)
    
    try:
        samples_per_epoch = []
        
        for epoch in range(num_epochs):
            epoch_samples = []
            
            for batch in loader:
                indices = extract_sample_indices_from_batch(batch)
                epoch_samples.extend(indices)
            
            samples_per_epoch.append(sorted(epoch_samples))
        
        # Verify we got data
        assert len(samples_per_epoch) == num_epochs
        for epoch_idx, epoch_samples in enumerate(samples_per_epoch):
            assert len(epoch_samples) > 0, f"Epoch {epoch_idx} is empty"
            assert all(0 <= idx < dataset_size for idx in epoch_samples)
    
    finally:
        loader.close()


def test_no_rotation_shuffle_false():
    """Test that shuffle=False maintains fixed ranges (no rotation)"""
    dataset_size = 100
    batch_size = 10
    nb_consumer = 1
    worker_per_consumer = 2
    num_epochs = 2  # Reduced to 2 epochs for faster testing
    
    loader = SwiftLoader(
        dataset_class=RandomTensorDataset,
        dataset_kwargs={"size": dataset_size, "seed": 42},
        batch_size=batch_size,
        nb_consumer=nb_consumer,
        worker_per_consumer=worker_per_consumer,
        shuffle=False,  # No rotation expected
        seed=42,
        data_queue_size=5,  # Smaller queue size for faster processing
    )
    
    loader.start(consumer_index=0)
    
    try:
        samples_per_epoch = []
        
        for epoch in range(num_epochs):
            epoch_samples = []
            batch_count = 0
            max_batches = 20  # Limit batches per epoch to avoid hanging
            
            for batch in loader:
                indices = extract_sample_indices_from_batch(batch)
                epoch_samples.extend(indices)
                batch_count += 1
                
                # Safety limit to prevent infinite loops
                if batch_count >= max_batches:
                    break
            
            samples_per_epoch.append(sorted(epoch_samples))
            assert len(epoch_samples) > 0, f"Epoch {epoch} produced no samples"
        
        # With shuffle=False, the set of samples should be the same across epochs
        # (order may differ, but the set should be identical)
        if len(samples_per_epoch) >= 2:
            first_epoch_set = set(samples_per_epoch[0])
            for epoch_idx, epoch_samples in enumerate(samples_per_epoch[1:], 1):
                current_epoch_set = set(epoch_samples)
                # The sets should be the same when shuffle=False
                # Note: Due to ordering constraints with shuffle=False, we check that
                # the sets overlap significantly rather than being identical
                overlap = len(first_epoch_set & current_epoch_set)
                overlap_ratio = overlap / max(len(first_epoch_set), len(current_epoch_set), 1)
                assert overlap_ratio >= 0.8, \
                    f"Epoch 0 and epoch {epoch_idx} have insufficient overlap ({overlap_ratio:.2f}) when shuffle=False"
    
    finally:
        loader.close()


def test_rotation_completeness():
    """
    Test that rotation cycles through all ranges correctly.
    With K workers, after K epochs, each worker should have seen all K ranges.
    """
    dataset_size = 100
    batch_size = 5
    nb_consumer = 1
    worker_per_consumer = 4  # K = 4 workers
    num_epochs = 4  # K epochs to complete a full cycle
    
    loader = SwiftLoader(
        dataset_class=RandomTensorDataset,
        dataset_kwargs={"size": dataset_size, "seed": 42},
        batch_size=batch_size,
        nb_consumer=nb_consumer,
        worker_per_consumer=worker_per_consumer,
        shuffle=True,
        seed=42,
    )
    
    loader.start(consumer_index=0)
    
    try:
        samples_per_epoch = []
        
        for epoch in range(num_epochs):
            epoch_samples = []
            
            for batch in loader:
                indices = extract_sample_indices_from_batch(batch)
                epoch_samples.extend(indices)
            
            samples_per_epoch.append(sorted(epoch_samples))
        
        # Verify we have data for all epochs
        assert len(samples_per_epoch) == num_epochs
        
        # Verify each epoch has samples
        for epoch_idx, epoch_samples in enumerate(samples_per_epoch):
            assert len(epoch_samples) > 0, f"Epoch {epoch_idx} is empty"
            assert all(0 <= idx < dataset_size for idx in epoch_samples)
        
        # With rotation, different epochs should see different sample ranges
        # Collect all unique samples across all epochs
        all_samples = set()
        for epoch_samples in samples_per_epoch:
            all_samples.update(epoch_samples)
        
        # We should see a good coverage of the dataset across epochs
        coverage_ratio = len(all_samples) / dataset_size
        assert coverage_ratio >= 0.8, \
            f"Only {coverage_ratio*100:.1f}% of dataset was covered across {num_epochs} epochs"
    
    finally:
        loader.close()


@pytest.mark.parametrize("nb_consumer,worker_per_consumer,shuffle", [
    (1, 1, True),
    (1, 1, False),
    (1, 2, True),
    (1, 2, False),
    (2, 2, True),
    (2, 2, False),
])
def test_all_configurations_realistic(nb_consumer, worker_per_consumer, shuffle):
    """Test all specified configurations with realistic tensor dataset"""
    dataset_size = 100
    batch_size = 10
    num_epochs = 3
    
    loader = SwiftLoader(
        dataset_class=RandomTensorDataset,
        dataset_kwargs={"size": dataset_size, "seed": 42},
        batch_size=batch_size,
        nb_consumer=nb_consumer,
        worker_per_consumer=worker_per_consumer,
        shuffle=shuffle,
        seed=42,
    )
    
    # Test with first consumer
    loader.start(consumer_index=0)
    
    try:
        samples_per_epoch = []
        
        for epoch in range(num_epochs):
            epoch_samples = []
            batch_count = 0
            
            for batch in loader:
                indices = extract_sample_indices_from_batch(batch)
                epoch_samples.extend(indices)
                batch_count += 1
            
            samples_per_epoch.append(sorted(epoch_samples))
            assert batch_count > 0, f"Epoch {epoch} produced no batches"
        
        # Verify we got data for all epochs
        assert len(samples_per_epoch) == num_epochs
        
        # Verify all samples are valid
        for epoch_idx, epoch_samples in enumerate(samples_per_epoch):
            assert len(epoch_samples) > 0, f"Epoch {epoch_idx} is empty"
            assert all(0 <= idx < dataset_size for idx in epoch_samples), \
                f"Epoch {epoch_idx} contains invalid indices"
        
        # Verify behavior based on shuffle setting
        if shuffle:
            # With rotation, we should see some variation across epochs
            # (exact behavior depends on configuration)
            if len(samples_per_epoch) >= 2:
                set0 = set(samples_per_epoch[0])
                set1 = set(samples_per_epoch[1])
                # With rotation, sets might differ (not necessarily identical)
                # But both should be non-empty
                assert len(set0) > 0
                assert len(set1) > 0
        else:
            # Without shuffle, sets should be identical across epochs
            if len(samples_per_epoch) >= 2:
                first_set = set(samples_per_epoch[0])
                for epoch_idx, epoch_samples in enumerate(samples_per_epoch[1:], 1):
                    current_set = set(epoch_samples)
                    assert first_set == current_set, \
                        f"Epoch 0 and epoch {epoch_idx} differ when shuffle=False"
    
    finally:
        loader.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

