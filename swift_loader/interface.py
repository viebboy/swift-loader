"""
interface.py: interfaces exposed to users
-----------------------------------------


* Copyright: 2023 Dat Tran
* Authors: Dat Tran
* Emails: hello@dats.bio
* Date: 2023-11-12
* Version: 0.0.1


This is part of the swift_loader package


License
-------
Apache 2.0

"""

from __future__ import annotations
from typing import Callable, Any
import traceback
import os
import dill
import time
import numpy as np
from swift_loader.utils import get_temp_file, get_default_collate_fn
from swift_loader.log import get_logger
from swift_loader.workers import WorkerManager


class SwiftLoader:
    def __init__(
        self,
        dataset_class: Callable,
        dataset_kwargs: dict,
        batch_size: int,
        nb_consumer: int,
        worker_per_consumer: int,
        shuffle: bool = True,
        **kwargs: dict,
    ):
        self._kwargs = self._verify_params(
            dataset_class,
            dataset_kwargs,
            batch_size,
            nb_consumer,
            worker_per_consumer,
            shuffle,
            **kwargs,
        )

        # dump dataset info to a temp file
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
    ):
        assert isinstance(nb_consumer, int) and nb_consumer > 0
        assert isinstance(worker_per_consumer, int) and worker_per_consumer > 0

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

        # handle logger
        if logger is None:
            logger = {"path": None, "stdout": False}

        if validate:
            # try construct dataset
            try:
                dataset = dataset_class(**dataset_kwargs)
            except BaseException as e:
                print(
                    f"Failed to construct dataset {dataset_class} from: {dataset_kwargs}"
                )
                traceback.print_exc()
                raise e

            # try collecting data
            try:
                batch_data = [dataset[i] for i in range(batch_size)]
            except BaseException as e:
                print(f"Failed to getitem [] from dataset {dataset_class}")
                traceback.print_exc()
                raise e

            # try collate_fn
            if collate_fn is None:
                collate_fn = get_default_collate_fn()
            try:
                batch_data = collate_fn(batch_data)
            except BaseException as e:
                print("Failed to collate minibatch data from dataset")
                traceback.print_exc()
                raise e

            # try to encode batch data
            try:
                batch_data = batch_encoder(batch_data)
            except BaseException as e:
                print("Failed to encode batch data")
                traceback.print_exc()
                raise e

            # try to decode batch data
            try:
                batch_data = batch_decoder(batch_data)
            except BaseException as e:
                print("Failed to decode batch data")
                traceback.print_exc()
                raise e

        if logger is not None:
            assert isinstance(logger, (str, dict))

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

    def _get_logger(self, consumer_index: int):
        if self._kwargs["logger"] is not None:
            config = self._kwargs["logger"].copy()
        else:
            config = {"path": None, "stdout": False}

        config["name"] = f"Dataloader-{consumer_index}"
        config["suffix"] = f"Dataloader-{consumer_index}"
        return get_logger(**config)

    def extend_clarifications(self, **clarifications: dict):
        clarifications["class_name"] = self.__class__.__name__
        return clarifications

    def print_log(self, message: str, **clarifications: dict):
        clarifications = [f"{key}={value}" for key, value in clarifications.items()]
        clarifications = ", ".join(clarifications)
        print(f"{message} ({clarifications})")

    def debug(self, message: str, **clarifications: dict):
        clarifications = self.extend_clarifications(**clarifications)
        if self._logger is None:
            self.print_log(message, **clarifications)
            return

        self._logger.debug(
            message,
            **clarifications,
        )

    def info(self, message: str, **clarifications: dict):
        clarifications = self.extend_clarifications(**clarifications)
        if self._logger is None:
            self.print_log(message, **clarifications)
            return

        self._logger.info(
            message,
            **clarifications,
        )

    def warning(self, message: str, **clarifications: dict):
        clarifications = self.extend_clarifications(**clarifications)
        if self._logger is None:
            self.print_log(message, **clarifications)
            return

        self._logger.warning(
            message,
            **clarifications,
        )

    def error(self, message: str, **clarifications: dict):
        clarifications = self.extend_clarifications(**clarifications)
        if self._logger is None:
            self.print_log(message, **clarifications)
            return

        self._logger.warning(
            message,
            **clarifications,
        )

    def start(
        self,
        consumer_index: int,
        device: Any = None,
        to_device: Callable = None,
    ):
        self._logger = self._get_logger(consumer_index)
        if self._is_started:
            self.error("start() has been called already", class_method="start")
            raise RuntimeError("start() has been called already")

        # create a worker manager
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

        # start the manager
        self._worker_manager.start(
            consumer_index=consumer_index, device=device, to_device=to_device
        )
        self._is_started = True
        self.debug("SwiftLoader started")

    def __enter__(self):
        return self._client.__enter__()

    def __exit__(self, *args):
        self.close()

    def __len__(self):
        if not self._is_started:
            self.error("start() has not been called yet", class_method="__len__")
            raise RuntimeError("start() has not been called yet")

        return len(self._worker_manager)

    def __iter__(self):
        return self._worker_manager.__iter__()

    def __next__(self):
        return next(self._worker_manager)

    def close(self):
        if not self._is_closed:
            self.debug("Trying to close WorkerManager", class_method="close")
            if self._worker_manager is not None:
                self._worker_manager.close()
            self._is_closed = True
            self.debug("SwiftLoader has closed")
