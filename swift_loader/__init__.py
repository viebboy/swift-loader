"""SwiftLoader - High-performance multiprocess data loader for machine learning.

SwiftLoader provides efficient data loading with multiprocessing support,
shared memory queues, and configurable batching strategies. It is designed
to maximize GPU utilization by overlapping data loading with model training.

Example:
    >>> from swift_loader import SwiftLoader
    >>> loader = SwiftLoader(
    ...     dataset_class=MyDataset,
    ...     dataset_kwargs={"data_path": "/path/to/data"},
    ...     batch_size=32,
    ...     nb_consumer=1,
    ...     worker_per_consumer=4,
    ...     shuffle=True
    ... )
    >>> loader.start(consumer_index=0, device="cuda:0")
    >>> for batch in loader:
    ...     # Process batch
    ...     pass
    >>> loader.close()
"""

from swift_loader.interface import SwiftLoader
from swift_loader.version import __version__

__all__ = ["SwiftLoader", "__version__"]
