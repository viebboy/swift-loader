"""Interface module for SwiftLoader - high-performance multiprocess data loader.

This module provides the main SwiftLoader class, which is the primary interface
for users to create and manage multiprocess data loading pipelines.

* Copyright: 2023 Dat Tran
* Authors: Dat Tran
* Emails: hello@dats.bio
* Date: 2023-11-12
* Version: 0.0.1

License
-------
Apache 2.0
"""

from __future__ import annotations

from typing import Any, Callable

import dill
import os
import time
import traceback

import numpy as np

from swift_loader.log import get_logger
from swift_loader.utils import get_default_collate_fn, get_temp_file
from swift_loader.workers import WorkerManager


class SwiftLoader:
    """High-performance multiprocess data loader for machine learning training.

    SwiftLoader provides efficient data loading with multiprocessing support,
    shared memory queues, and configurable batching strategies. It is designed
    to maximize GPU utilization by overlapping data loading with model training.

    Attributes:
        _kwargs: Internal dictionary storing configuration parameters.
        _dataset_file: Path to temporary file containing serialized dataset info.
        _is_started: Flag indicating whether the loader has been started.
        _is_closed: Flag indicating whether the loader has been closed.
        _logger: Logger instance for logging operations.
        _worker_manager: WorkerManager instance managing worker processes.

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

    def __init__(
        self,
        dataset_class: Callable,
        dataset_kwargs: dict,
        batch_size: int,
        nb_consumer: int,
        worker_per_consumer: int,
        shuffle: bool = True,
        **kwargs: dict,
    ) -> None:
        """Initialize SwiftLoader.

        Args:
            dataset_class: Callable class or function that creates the dataset.
                Must be constructible with dataset_kwargs.
            dataset_kwargs: Dictionary of keyword arguments passed to
                dataset_class constructor.
            batch_size: Number of samples per batch.
            nb_consumer: Number of consumer processes (typically 1 per GPU).
            worker_per_consumer: Number of worker processes per consumer.
            shuffle: Whether to shuffle data. Defaults to True.
            **kwargs: Additional optional parameters:
                seed: Random seed for shuffling. Defaults to current timestamp.
                data_queue_size: Maximum size of data queue. Defaults to 10.
                message_queue_size: Maximum size of message queue. Defaults to 10.
                collate_fn: Function to collate batch samples. If None, uses
                    default collate function.
                logger: Logger configuration (dict or str path). Defaults to None.
                batch_encoder: Function to encode batches. Defaults to dill.dumps.
                batch_decoder: Function to decode batches. Defaults to dill.loads.
                nearby_shuffle: Window size for nearby shuffling. Defaults to
                    5 * batch_size.
                benchmark: Whether to enable benchmarking. Defaults to True.
                benchmark_file: Path to benchmark log file. Defaults to None.
                validate: Whether to validate dataset during initialization.
                    Defaults to True.
                multithread: Whether to use multithreading. Defaults to False.
                max_nb_batch_on_device: Maximum number of batches to keep on
                    device. Defaults to 1.
                max_buffer_size: Maximum buffer size. Defaults to calculated value.

        Raises:
            AssertionError: If nb_consumer or worker_per_consumer are invalid.
            ValueError: If dataset validation fails.
            RuntimeError: If dataset construction or data access fails.
        """
        self._kwargs = self._verify_params(
            dataset_class,
            dataset_kwargs,
            batch_size,
            nb_consumer,
            worker_per_consumer,
            shuffle,
            **kwargs,
        )

        # Dump dataset info to a temp file for worker processes
        self._dataset_file = get_temp_file()
        with open(self._dataset_file, "wb") as fid:
            info = {
                "dataset_class": dataset_class,
                "dataset_kwargs": dataset_kwargs,
                "shuffle": self._kwargs["shuffle"],
                "nb_worker": self._kwargs["nb_worker"],
                "batch_size": self._kwargs["batch_size"],
                "nearby_shuffle": self._kwargs["nearby_shuffle"],
                "collate_fn": self._kwargs["collate_fn"],
                "batch_encoder": self._kwargs["batch_encoder"],
                "batch_decoder": self._kwargs["batch_decoder"],
                "benchmark": self._kwargs["benchmark"],
                "benchmark_file": self._kwargs["benchmark_file"],
                "data_queue_size": self._kwargs["data_queue_size"],
                "message_queue_size": self._kwargs["message_queue_size"],
                "logger": self._kwargs["logger"],
                "seed": self._kwargs["seed"],
            }
            dill.dump(info, fid, recurse=True)

        self._is_started = False
        self._is_closed = False
        self._logger = None
        self._worker_manager = None

    def _verify_params(
        self,
        dataset_class: Callable,
        dataset_kwargs: dict,
        batch_size: int,
        nb_consumer: int,
        worker_per_consumer: int,
        shuffle: bool,
        **kwargs: dict,
    ) -> dict:
        """Verify and process initialization parameters.

        Args:
            dataset_class: Dataset class to verify.
            dataset_kwargs: Dataset keyword arguments.
            batch_size: Batch size.
            nb_consumer: Number of consumers.
            worker_per_consumer: Workers per consumer.
            shuffle: Whether to shuffle.
            **kwargs: Additional keyword arguments.

        Returns:
            Dictionary of verified and processed parameters.

        Raises:
            AssertionError: If parameters are invalid.
            ValueError: If dataset validation fails.
            RuntimeError: If dataset operations fail.
        """
        assert (
            isinstance(nb_consumer, int) and nb_consumer > 0
        ), "nb_consumer must be a positive integer"
        assert (
            isinstance(worker_per_consumer, int) and worker_per_consumer > 0
        ), "worker_per_consumer must be a positive integer"

        seed_number = kwargs.get("seed", int(time.time() * 1000))
        data_queue_size = kwargs.get("data_queue_size", 10)
        message_queue_size = kwargs.get("message_queue_size", 10)
        collate_fn = kwargs.get("collate_fn", None)
        logger = kwargs.get("logger", None)
        batch_encoder = kwargs.get("batch_encoder", dill.dumps)
        batch_decoder = kwargs.get("batch_decoder", dill.loads)
        nearby_shuffle = kwargs.get("nearby_shuffle", 5 * batch_size)
        benchmark = kwargs.get("benchmark", True)
        benchmark_file = kwargs.get("benchmark_file", None)
        validate = kwargs.get("validate", True)
        multithread = kwargs.get("multithread", False)
        max_nb_batch_on_device = kwargs.get("max_nb_batch_on_device", 1)
        max_buffer_size = kwargs.get(
            "max_buffer_size",
            max(2, int(np.ceil(data_queue_size / worker_per_consumer))),
        )

        # Handle logger configuration
        if logger is None:
            logger = {"path": None, "stdout": False}

        if validate:
            # Try to construct dataset
            try:
                dataset = dataset_class(**dataset_kwargs)
            except BaseException as e:
                print(
                    f"Failed to construct dataset {dataset_class} from: {dataset_kwargs}"
                )
                traceback.print_exc()
                raise e

            # Try collecting data
            try:
                batch_data = [dataset[i] for i in range(batch_size)]
            except BaseException as e:
                print(f"Failed to getitem [] from dataset {dataset_class}")
                traceback.print_exc()
                raise e

            # Try collate_fn
            if collate_fn is None:
                collate_fn = get_default_collate_fn()
            try:
                batch_data = collate_fn(batch_data)
            except BaseException as e:
                print("Failed to collate minibatch data from dataset")
                traceback.print_exc()
                raise e

            # Try to encode batch data
            try:
                batch_data = batch_encoder(batch_data)
            except BaseException as e:
                print("Failed to encode batch data")
                traceback.print_exc()
                raise e

            # Try to decode batch data
            try:
                batch_data = batch_decoder(batch_data)
            except BaseException as e:
                print("Failed to decode batch data")
                traceback.print_exc()
                raise e

        if logger is not None:
            assert isinstance(
                logger, (str, dict)
            ), "logger must be a string path or dictionary"

        if isinstance(logger, str):
            logger = {"path": logger}
            base_path = os.path.dirname(logger["path"])
            if not os.path.exists(base_path):
                os.makedirs(base_path, exist_ok=True)

        return {
            "batch_size": batch_size,
            "nb_consumer": nb_consumer,
            "worker_per_consumer": worker_per_consumer,
            "nb_worker": nb_consumer * worker_per_consumer,
            "seed": seed_number,
            "data_queue_size": data_queue_size,
            "message_queue_size": message_queue_size,
            "max_nb_batch_on_device": max_nb_batch_on_device,
            "collate_fn": collate_fn,
            "logger": logger,
            "shuffle": shuffle,
            "batch_encoder": batch_encoder,
            "batch_decoder": batch_decoder,
            "nearby_shuffle": nearby_shuffle,
            "benchmark": benchmark,
            "benchmark_file": benchmark_file,
            "multithread": multithread,
            "max_buffer_size": max_buffer_size,
        }

    def _get_logger(self, consumer_index: int) -> Any:
        """Get logger instance for a specific consumer.

        Args:
            consumer_index: Index of the consumer.

        Returns:
            Logger instance.
        """
        if self._kwargs["logger"] is not None:
            config = self._kwargs["logger"].copy()
        else:
            config = {"path": None, "stdout": False}

        config["name"] = f"Dataloader-{consumer_index}"
        config["suffix"] = f"Dataloader-{consumer_index}"
        return get_logger(**config)

    def extend_clarifications(self, **clarifications: dict) -> dict:
        """Extend clarification dictionary with class name.

        Args:
            **clarifications: Additional clarification key-value pairs.

        Returns:
            Dictionary with class name added.
        """
        clarifications["class_name"] = self.__class__.__name__
        return clarifications

    def print_log(self, message: str, **clarifications: dict) -> None:
        """Print log message when logger is not available.

        Args:
            message: Log message.
            **clarifications: Additional context information.
        """
        clarifications_list = [
            f"{key}={value}" for key, value in clarifications.items()
        ]
        clarifications_str = ", ".join(clarifications_list)
        print(f"{message} ({clarifications_str})")

    def debug(self, message: str, **clarifications: dict) -> None:
        """Log debug message.

        Args:
            message: Debug message.
            **clarifications: Additional context information.
        """
        clarifications = self.extend_clarifications(**clarifications)
        if self._logger is None:
            self.print_log(message, **clarifications)
            return

        self._logger.debug(message, **clarifications)

    def info(self, message: str, **clarifications: dict) -> None:
        """Log info message.

        Args:
            message: Info message.
            **clarifications: Additional context information.
        """
        clarifications = self.extend_clarifications(**clarifications)
        if self._logger is None:
            self.print_log(message, **clarifications)
            return

        self._logger.info(message, **clarifications)

    def warning(self, message: str, **clarifications: dict) -> None:
        """Log warning message.

        Args:
            message: Warning message.
            **clarifications: Additional context information.
        """
        clarifications = self.extend_clarifications(**clarifications)
        if self._logger is None:
            self.print_log(message, **clarifications)
            return

        self._logger.warning(message, **clarifications)

    def error(self, message: str, **clarifications: dict) -> None:
        """Log error message.

        Args:
            message: Error message.
            **clarifications: Additional context information.
        """
        clarifications = self.extend_clarifications(**clarifications)
        if self._logger is None:
            self.print_log(message, **clarifications)
            return

        self._logger.error(message, **clarifications)

    def start(
        self,
        consumer_index: int,
        device: Any = None,
        to_device: Callable = None,
    ) -> None:
        """Start the data loader.

        Args:
            consumer_index: Index of the consumer (typically 0 for single GPU).
            device: Target device (e.g., "cuda:0" or torch.device).
            to_device: Function to move data to device. If None, uses default.

        Raises:
            RuntimeError: If start() has already been called.
        """
        self._logger = self._get_logger(consumer_index)
        if self._is_started:
            self.error("start() has been called already", class_method="start")
            raise RuntimeError("start() has been called already")

        # Create a worker manager
        self._worker_manager = WorkerManager(
            dataset_file=self._dataset_file,
            nb_consumer=self._kwargs["nb_consumer"],
            worker_per_consumer=self._kwargs["worker_per_consumer"],
            shuffle=self._kwargs["shuffle"],
            seed=self._kwargs["seed"],
            multithread=self._kwargs["multithread"],
            data_queue_size=self._kwargs["data_queue_size"],
            max_nb_batch_on_device=self._kwargs["max_nb_batch_on_device"],
            max_buffer_size=self._kwargs["max_buffer_size"],
        )

        # Start the manager
        self._worker_manager.start(
            consumer_index=consumer_index, device=device, to_device=to_device
        )
        self._is_started = True
        self.debug("SwiftLoader started")

    def __enter__(self) -> SwiftLoader:
        """Context manager entry.

        Returns:
            Self instance.
        """
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit.

        Args:
            *args: Exception information (if any).
        """
        self.close()

    def __len__(self) -> int:
        """Get the number of batches per epoch.

        Returns:
            Number of batches.

        Raises:
            RuntimeError: If start() has not been called.
        """
        if not self._is_started:
            self.error("start() has not been called yet", class_method="__len__")
            raise RuntimeError("start() has not been called yet")

        return len(self._worker_manager)

    def __iter__(self) -> SwiftLoader:
        """Return iterator for the data loader.

        Returns:
            Iterator instance.
        """
        return self._worker_manager.__iter__()

    def __next__(self) -> Any:
        """Get next batch.

        Returns:
            Next batch of data.

        Raises:
            StopIteration: When epoch ends.
        """
        return next(self._worker_manager)

    def close(self) -> None:
        """Close the data loader and clean up resources."""
        if not self._is_closed:
            self.debug("Trying to close WorkerManager", class_method="close")
            if self._worker_manager is not None:
                self._worker_manager.close()
            self._is_closed = True
            self.debug("SwiftLoader has closed")
