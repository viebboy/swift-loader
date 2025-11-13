"""
Direct tests for Worker rotation logic

These tests directly test the Worker's prepare_dataset and collect_minibatch
methods to verify rotation behavior.
"""

import pytest
import numpy as np
import dill
import tempfile
import os
from swift_loader.workers import Worker
from swift_loader.utils import Property, shuffle_indices
from multiprocessing import get_context
import time

CTX = get_context("spawn")


class MockDataset:
    """Mock dataset for testing"""
    
    def __init__(self, size=100):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return idx


class MockWorker(Worker):
    """Mock Worker that allows direct testing of rotation logic"""
    
    def __init__(self, dataset_file, nb_consumer, worker_per_consumer, consumer_index, worker_index):
        # Don't call super().__init__ to avoid process creation
        self.dataset_file = dataset_file
        self.nb_consumer = nb_consumer
        self.worker_per_consumer = worker_per_consumer
        self.consumer_index = consumer_index
        self.worker_index = worker_index
        self.name = f"Worker-{worker_index}"
        self.side = "child"
        
        # Initialize data property
        self.data = Property()
        self.logger = None
        self.seed = 42  # Default seed
        
        # Initialize attributes needed by prepare_dataset and child_task_guard
        self.benchmark_logger = None
        self.is_child_clean = False
        
        # Mock pipes (won't be used but needed to avoid AttributeError)
        from multiprocessing import Pipe
        self.back_read_pipe, self.back_write_pipe = Pipe()
    
    def debug(self, message, **kwargs):
        """Mock debug method"""
        pass
    
    def get_logger(self, **kwargs):
        """Mock logger"""
        return None
    
    def notify_parent(self, **message):
        """Mock notify_parent - do nothing in tests"""
        pass
    
    def clean_child(self):
        """Mock clean_child - do nothing in tests"""
        pass


def create_dataset_file(dataset_class, dataset_kwargs, shuffle, seed=42):
    """Create a temporary dataset file for testing"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
    temp_file.close()
    
    info = {
        "dataset_class": dataset_class,
        "dataset_kwargs": dataset_kwargs,
        "shuffle": shuffle,
        "nb_worker": None,  # Will be set based on nb_consumer and worker_per_consumer
        "batch_size": 10,
        "nearby_shuffle": 50,
        "collate_fn": lambda x: x,  # Simple collate that returns list as-is
        "batch_encoder": lambda x: x,  # No encoding for testing
        "batch_decoder": lambda x: x,  # No decoding for testing
        "benchmark": False,
        "benchmark_file": None,
        "data_queue_size": 10,
        "message_queue_size": 10,
        "logger": None,
        "seed": seed,
    }
    
    with open(temp_file.name, "wb") as f:
        dill.dump(info, f, recurse=True)
    
    return temp_file.name


def test_worker_prepare_dataset_shuffle_true():
    """Test that prepare_dataset sets up correct initial ranges when shuffle=True"""
    dataset_size = 100
    nb_consumer = 2
    worker_per_consumer = 2
    nb_worker = nb_consumer * worker_per_consumer
    worker_size = int(np.ceil(dataset_size / nb_worker))
    
    dataset_file = create_dataset_file(MockDataset, {"size": dataset_size}, shuffle=True)
    
    try:
        # Test each worker
        for consumer_idx in range(nb_consumer):
            for worker_idx_in_consumer in range(worker_per_consumer):
                worker_index = consumer_idx * worker_per_consumer + worker_idx_in_consumer
                
                worker = MockWorker(
                    dataset_file, nb_consumer, worker_per_consumer, consumer_idx, worker_index
                )
                
                # Load kwargs
                with open(dataset_file, "rb") as f:
                    kwargs = dill.load(f)
                kwargs["nb_worker"] = nb_worker
                
                # Call prepare_dataset
                worker.prepare_dataset(**kwargs)
                
                # Verify initial range
                expected_start = worker_index * worker_size
                expected_stop = min(expected_start + worker_size, dataset_size)
                
                assert worker.data.start == expected_start, \
                    f"Worker {worker_index}: expected start={expected_start}, got {worker.data.start}"
                assert worker.data.stop == expected_stop, \
                    f"Worker {worker_index}: expected stop={expected_stop}, got {worker.data.stop}"
                assert len(worker.data.indices) == (expected_stop - expected_start), \
                    f"Worker {worker_index}: expected {expected_stop - expected_start} indices, got {len(worker.data.indices)}"
                
                # Verify indices are in the correct range
                assert all(expected_start <= idx < expected_stop for idx in worker.data.indices), \
                    f"Worker {worker_index}: indices out of range"
                
                # Verify nb_worker and worker_size are stored
                assert hasattr(worker.data, 'nb_worker')
                assert hasattr(worker.data, 'worker_size')
                assert worker.data.nb_worker == nb_worker
                assert worker.data.worker_size == worker_size
    finally:
        os.unlink(dataset_file)


def test_worker_rotation_shuffle_true():
    """Test that workers rotate through ranges correctly when shuffle=True"""
    dataset_size = 100
    nb_consumer = 1
    worker_per_consumer = 4
    nb_worker = nb_consumer * worker_per_consumer
    worker_size = int(np.ceil(dataset_size / nb_worker))
    num_epochs = nb_worker  # Complete a full rotation cycle
    
    dataset_file = create_dataset_file(MockDataset, {"size": dataset_size}, shuffle=True)
    
    try:
        # Test worker 0
        worker_index = 0
        consumer_index = 0
        
        worker = MockWorker(
            dataset_file, nb_consumer, worker_per_consumer, consumer_index, worker_index
        )
        
        # Load kwargs
        with open(dataset_file, "rb") as f:
            kwargs = dill.load(f)
        kwargs["nb_worker"] = nb_worker
        
        # Prepare dataset
        worker.prepare_dataset(**kwargs)
        
        # Initialize epoch_start for timing (required by collect_minibatch)
        worker.data.epoch_start = time.perf_counter()
        
        # Track ranges across epochs
        ranges_seen = []
        
        # Simulate multiple epochs
        for epoch in range(num_epochs):
            # Record current range at start of epoch
            ranges_seen.append((worker.data.start, worker.data.stop))
            
            # Simulate processing all batches in epoch
            batches_in_epoch = 0
            num_batches = worker.data.nb_batch
            for batch_idx in range(num_batches):
                # Collect a minibatch (this will trigger rotation at epoch end on last batch)
                batch = worker.collect_minibatch()
                batches_in_epoch += 1
                
                # Verify batch contains indices in the range that was set at epoch start
                epoch_start, epoch_stop = ranges_seen[-1]
                assert all(epoch_start <= idx < epoch_stop for idx in batch), \
                    f"Epoch {epoch}: batch indices {batch} out of range [{epoch_start}, {epoch_stop})"
            
            # After processing all batches, rotation should have been triggered
            # Verify we processed the expected number of batches
            assert batches_in_epoch == num_batches, \
                f"Expected {num_batches} batches, processed {batches_in_epoch}"
        
        # Verify rotation: each epoch should have a different range
        range_starts = [r[0] for r in ranges_seen]
        unique_starts = set(range_starts)
        
        # After nb_worker epochs, we should have seen nb_worker different ranges
        assert len(unique_starts) == nb_worker, \
            f"Expected {nb_worker} different ranges, got {len(unique_starts)}. Ranges: {ranges_seen}"
        
        # Verify the ranges form a complete cycle
        # Worker 0 should see ranges: 0, 1, 2, ..., nb_worker-1 (rotating)
        # In epoch 0: range 0, epoch 1: range 1, ..., epoch nb_worker-1: range nb_worker-1
        expected_starts = [((worker_index + epoch) % nb_worker) * worker_size 
                          for epoch in range(num_epochs)]
        assert range_starts == expected_starts, \
            f"Expected ranges starting at {expected_starts}, got {range_starts}"
        
    finally:
        os.unlink(dataset_file)


def test_worker_no_rotation_shuffle_false():
    """Test that workers maintain fixed ranges when shuffle=False"""
    dataset_size = 100
    nb_consumer = 2
    worker_per_consumer = 2
    
    dataset_file = create_dataset_file(MockDataset, {"size": dataset_size}, shuffle=False)
    
    try:
        # Test worker 0
        worker_index = 0
        consumer_index = 0
        
        worker = MockWorker(
            dataset_file, nb_consumer, worker_per_consumer, consumer_index, worker_index
        )
        
        # Load kwargs
        with open(dataset_file, "rb") as f:
            kwargs = dill.load(f)
        kwargs["nb_worker"] = nb_consumer * worker_per_consumer
        
        # Prepare dataset
        worker.prepare_dataset(**kwargs)
        
        # Initialize epoch_start for timing (required by collect_minibatch)
        worker.data.epoch_start = time.perf_counter()
        
        # Record initial indices
        initial_indices_set = set(worker.data.indices)
        
        # When shuffle=False, start and stop are not set (only used for shuffle=True)
        # So we just verify that indices remain the same across epochs
        
        # Simulate multiple epochs
        num_epochs = 3
        for epoch in range(num_epochs):
            # Process all batches
            num_batches = worker.data.nb_batch
            for batch_idx in range(num_batches):
                batch = worker.collect_minibatch()
            
            # After epoch, with shuffle=False, there's no rotation
            # So indices should remain the same set (order may differ due to shuffling within range)
            # But the set of indices should be the same
            
        # Verify indices haven't changed (no rotation)
        # Note: with shuffle=False, indices might be reordered but the set should be the same
        final_indices_set = set(worker.data.indices)
        assert initial_indices_set == final_indices_set, \
            f"Indices changed when shuffle=False. Initial: {sorted(initial_indices_set)}, Final: {sorted(final_indices_set)}"
        
    finally:
        os.unlink(dataset_file)


@pytest.mark.parametrize("nb_consumer,worker_per_consumer", [
    (1, 1),
    (1, 2),
    (2, 2),
])
def test_rotation_all_configurations(nb_consumer, worker_per_consumer):
    """Test rotation for all specified configurations"""
    dataset_size = 100
    nb_worker = nb_consumer * worker_per_consumer
    worker_size = int(np.ceil(dataset_size / nb_worker))
    num_epochs = nb_worker  # Complete rotation cycle
    
    dataset_file = create_dataset_file(MockDataset, {"size": dataset_size}, shuffle=True)
    
    try:
        # Test each worker
        for consumer_idx in range(nb_consumer):
            for worker_idx_in_consumer in range(worker_per_consumer):
                worker_index = consumer_idx * worker_per_consumer + worker_idx_in_consumer
                
                worker = MockWorker(
                    dataset_file, nb_consumer, worker_per_consumer, consumer_idx, worker_index
                )
                
                # Load kwargs
                with open(dataset_file, "rb") as f:
                    kwargs = dill.load(f)
                kwargs["nb_worker"] = nb_worker
                
                # Prepare dataset
                worker.prepare_dataset(**kwargs)
                
                # Initialize epoch_start for timing (required by collect_minibatch)
                worker.data.epoch_start = time.perf_counter()
                
                # Track ranges across epochs
                ranges_seen = []
                
                # Simulate epochs
                for epoch in range(num_epochs):
                    # Record range at start of epoch
                    ranges_seen.append((worker.data.start, worker.data.stop))
                    
                    # Process all batches in this epoch
                    num_batches = worker.data.nb_batch
                    for batch_idx in range(num_batches):
                        batch = worker.collect_minibatch()
                
                # Verify we saw different ranges
                range_starts = [r[0] for r in ranges_seen]
                unique_starts = set(range_starts)
                
                # Each worker should see nb_worker different ranges over nb_worker epochs
                assert len(unique_starts) == nb_worker, \
                    f"Worker {worker_index}: Expected {nb_worker} different ranges, got {len(unique_starts)}"
                
                # Verify the ranges are correct
                # Worker i should see ranges starting at: (i+0)*worker_size, (i+1)*worker_size, ..., (i+nb_worker-1)*worker_size (mod nb_worker)
                expected_starts = [((worker_index + epoch) % nb_worker) * worker_size 
                                 for epoch in range(num_epochs)]
                assert range_starts == expected_starts, \
                    f"Worker {worker_index}: Expected ranges {expected_starts}, got {range_starts}"
    
    finally:
        os.unlink(dataset_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

