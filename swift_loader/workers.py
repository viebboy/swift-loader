"""
workers.py: multiprocessing implementation
------------------------------------------


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
from typing import Any, Callable
from collections import deque
import dill
import multiprocessing
import threading
import time
import numpy as np
import traceback
import random
from queue import Queue

from swift_loader.utils import (
    BenchmarkProperty,
    ThroughputMeasure,
    Property,
    shuffle_indices,
    get_default_collate_fn,
    get_default_to_device,
    TimeMeasure,
    DummyTimeMeasure,
)
from swift_loader.log import get_logger
from swift_loader.mem_queue import SharedMemoryQueue
from swift_loader.signals import PARENT_MESSAGE, CHILD_MESSAGE

CTX = multiprocessing.get_context("spawn")


def signal_handler(
    nb_batch: int,
    read_pipe: CTX.Connection,
    write_pipe: CTX.Connection,
    data_queue: SharedMemoryQueue,
    close_event: threading.Event,
    parent: Worker,
    max_pipe_size: int,
):
    """
    Thread (run in child process) to communicate with parent process

    It first sends metadata related to the part of dataset handled by this worker
    and waits for PARENT_MESSAGE.ACK from parent

    Then it starts to pull out data from data_queue and send data info to parent

    After an epoch finishes, it sends CHILD_MESSAGE.EPOCH_END to parent
    """

    try:
        # telcom with parent here
        metadata_sent = False

        # list of samples sent to parent but not yet removed from queue
        sent_samples = []

        # counter of epoch and minibatches
        nb_sent_batch = 0
        nb_sent_epoch = 0

        while True:
            # check close event
            if close_event.is_set():
                return

            """
            Sending metadata about the number of minibatches
            """
            if not metadata_sent:
                # Get current nb_batch (may change after rotation)
                current_nb_batch = parent.data.nb_batch if hasattr(parent, 'data') and hasattr(parent.data, 'nb_batch') else nb_batch
                write_pipe.send(
                    {
                        "type": CHILD_MESSAGE.METADATA.value,
                        "nb_batch": current_nb_batch,
                        "shared_memory": data_queue.name(),
                    }
                )

                # wait for ACK
                while True:
                    if read_pipe.poll():
                        resp_code = read_pipe.recv()
                        response = PARENT_MESSAGE.decode(resp_code)
                        if response == PARENT_MESSAGE.ACK:
                            parent.debug(
                                "Receive metadata ACK from parent",
                                method="signal_handler() thread",
                            )
                            break
                        elif response == PARENT_MESSAGE.TERMINATE:
                            parent.debug(
                                "Receive terminate signal from parent",
                                source="signal_handler() thread",
                            )
                            close_event.set()
                            return
                        else:
                            raise ValueError(
                                f"Unexpected response from parent: {response}",
                                source="workers.signal_handler",
                            )

                metadata_sent = True

            """
            Sending minibatches
            """
            get_called = False
            if len(sent_samples) <= max_pipe_size:
                # if pipe is not too full
                # can send
                shm_indices = data_queue.get()

                if shm_indices is None:
                    # queue has nothing
                    time.sleep(0.001)
                else:
                    # send to parent
                    write_pipe.send(shm_indices)

                    # keep track of what has been sent
                    sent_samples.append(shm_indices)

                    # increase counter
                    nb_sent_batch += 1

                    # Get current nb_batch dynamically (may change after rotation)
                    current_nb_batch = parent.data.nb_batch if hasattr(parent, 'data') and hasattr(parent.data, 'nb_batch') else nb_batch
                    if nb_sent_batch == current_nb_batch:
                        # end of epoch
                        nb_sent_batch = 0
                        nb_sent_epoch += 1

                        # need to send this to parent
                        write_pipe.send(CHILD_MESSAGE.EPOCH_END.value)
                        parent.debug(
                            "Epoch has ended, sending signal to parent",
                            method="signal_handler() thread",
                        )

                        # and append None to sent_samples
                        sent_samples.append(None)
                get_called = True

            # pipe is full, need to flush
            if read_pipe.poll():
                # flush
                resp_code = read_pipe.recv()
                response = PARENT_MESSAGE.decode(resp_code)
                if response == PARENT_MESSAGE.ACK:
                    # remove the sent sample from queue
                    indices = sent_samples.pop(0)
                    if indices is not None:
                        # if not None, remove them from queue
                        # if None, it is just coming from the ack of epoch end
                        data_queue.remove(indices[0], indices[1])

                elif response == PARENT_MESSAGE.TERMINATE:
                    parent.debug(
                        "Receive terminate signal from parent",
                        method="signal_handler() thread",
                    )
                    close_event.set()
                    return
                else:
                    raise ValueError(
                        f"Unexpected response from parent ({parent._name}): {response}",
                    )
            else:
                if not get_called:
                    time.sleep(0.001)

    except BaseException as error:
        # tell parent
        traceback_content = traceback.format_exc()
        write_pipe.send(
            {
                "type": CHILD_MESSAGE.TERMINATE.value,
                "traceback": traceback_content,
                "error": str(error),
            }
        )
        # then set close event
        close_event.set()

        # log and print traceback
        parent.error(str(error), method="signal_handler() thread")
        traceback.print_exc()


def child_task_guard(class_method: Callable):
    """
    Decorator to safe guard a method in child process
    """

    def execute_with_guard(self, *args, **kwargs):
        try:
            return class_method(self, *args, **kwargs)
        except BaseException as error:
            # get traceback and log
            traceback_content = traceback.format_exc()
            self.error(
                str(error),
                class_method=class_method.__name__,
                traceback=traceback_content,
            )

            # print out traceback in case no logger
            print(traceback_content)

            # notify parent first, clean after
            self.notify_parent(
                type=CHILD_MESSAGE.TERMINATE.value,
                error=str(error),
                worker_index=self.worker_index,
                class_method=class_method.__name__,
                class_name=self.__class__.__name__,
                traceback=traceback_content,
            )

            # clean the child
            self.clean_child()

    return execute_with_guard


def parent_task_guard(class_method: Callable):
    """
    Decorator to safe guard a method run in parent process
    """

    def execute_with_guard(self, *args, **kwargs):
        try:
            return class_method(self, *args, **kwargs)
        except BaseException as error:
            # get traceback and log
            traceback_content = traceback.format_exc()
            if self.logger is not None:
                self.error(
                    str(error),
                    class_method=class_method.__name__,
                    traceback=traceback_content,
                )

            # print out traceback in case no logger
            print(traceback_content)

            # calling close to clean up
            self.close()

    return execute_with_guard


def manager_task_guard(class_method: Callable):
    """
    Decorator to safe guard a method in WorkerManager
    """

    def execute_with_guard(self, *args, **kwargs):
        try:
            return class_method(self, *args, **kwargs)
        except BaseException as error:
            # get traceback and log
            traceback_content = traceback.format_exc()
            if self.logger is not None:
                self.error(
                    str(error),
                    class_method=class_method.__name__,
                    traceback=traceback_content,
                )

            # print out traceback in case no logger
            print(traceback_content)

            # calling close to clean up
            self.close()
            raise error

    return execute_with_guard


class OrchestrateSingleWorker(threading.Thread):
    def __init__(
        self,
        shuffle: bool,
        thread_index: int,
        worker: Worker,
        data_queue: DataQueue,
        close_event: threading.Event,
        epoch_end: threading.Event,
        parent: WorkerManager,
        max_buffer_size: int,
    ):
        super().__init__()
        self.shuffle = shuffle
        self.thread_index = thread_index
        self.worker = worker
        self.data_queue = data_queue
        self.close_event = close_event
        self.epoch_end = epoch_end
        self.parent = parent
        self.max_buffer_size = max_buffer_size
        if max_buffer_size > 0:
            self.use_buffer = True
        else:
            self.use_buffer = False

    def run_with_buffer(self):
        try:
            self.parent.debug(
                "Starting orchestration thread",
                thread_index=self.thread_index,
                worker_index=self.worker.index(),
                shuffle=self.shuffle,
                class_name="OrchestrateSingleWorker",
                class_method="run_with_buffer",
            )
            # buffer to hold intermediate batches
            buffer = deque()
            while True:
                if self.close_event.is_set():
                    # simply return if close event
                    self.parent.debug(
                        "Received close event from parent. Terminating thread now",
                        thread_index=self.thread_index,
                        worker_index=self.worker.index(),
                        shuffle=self.shuffle,
                        class_name="OrchestrateSingleWorker",
                        class_method="run_with_buffer",
                    )
                    return

                # parent will unset epoch end event
                # so when it is unset, then we can pull
                # the idea is that we need to wait for all
                # workers to finish their epoch before starting new ones
                if not self.epoch_end.is_set():
                    if len(buffer) > 0:
                        # if there are items in buffer, try to put to data queue
                        if buffer[0] is CHILD_MESSAGE.EPOCH_END:
                            # set epoch end so parent knows
                            # and we don't put batch into data_queue until the flag
                            # is cleared by parent
                            # we dont want to set after putting data into the queue
                            # because parent might pull too fast and check this flag
                            # which might not be set
                            self.epoch_end.set()

                            # if epoch end, we need to put until successful
                            # EPOCH_END bypasses turn ordering, so should succeed immediately
                            if self.close_event.is_set():
                                self.parent.debug(
                                    "Received close event from parent. Terminating thread now",
                                    thread_index=self.thread_index,
                                    worker_index=self.worker.index(),
                                    shuffle=self.shuffle,
                                    class_name="OrchestrateSingleWorker",
                                    class_method="run_with_buffer",
                                )
                                return
                            success = self.data_queue.put(
                                CHILD_MESSAGE.EPOCH_END, self.thread_index
                            )
                            if success:
                                buffer.popleft()
                            else:
                                # Should not happen for EPOCH_END, but handle it
                                time.sleep(0.001)

                        else:
                            if self.shuffle:
                                # if shuffling, we dont need to wait until successful
                                success = self.data_queue.put(
                                    buffer[0], self.thread_index
                                )
                                if success:
                                    # if successful, remove from buffer
                                    buffer.popleft()
                            else:
                                # For shuffle=False, put() returns False if not our turn or queue full
                                # Use condition variable to wait efficiently instead of busy-waiting
                                success = self.data_queue.put(
                                    buffer[0], self.thread_index
                                )
                                if not success:
                                    # Not our turn or queue full - wait on condition
                                    # The queue will notify us when turn changes or space available
                                    with self.data_queue.condition:
                                        # Check again after acquiring lock
                                        if self.close_event.is_set():
                                            return
                                        # Wait with timeout to periodically check close_event
                                        self.data_queue.condition.wait(timeout=0.1)
                                    # After wait, try again in next iteration
                                else:
                                    buffer.popleft()

                # we pull data from child process and put into buffer
                # regardless of whether epoch has ended
                # that's the idea of using buffer
                if len(buffer) < self.max_buffer_size:
                    data = self.worker.get()

                    if data is not None:
                        buffer.append(data)
                else:
                    # if buffer is also full, then we need to wait
                    time.sleep(0.001)

        except BaseException as error:
            # get traceback content
            traceback_content = traceback.format_exc()
            # log and print
            self.parent.error(
                str(error),
                thread_index=self.thread_index,
                worker_index=self.worker.index(),
                shuffle=self.shuffle,
                class_name="OrchestrateSingleWorker",
                class_method="run_with_buffer",
                traceback=traceback_content,
            )
            print(traceback_content)

            # set event so parent knows
            self.close_event.set()

            # then raise error
            raise error

    def run_without_buffer(self):
        try:
            self.parent.debug(
                "Starting orchestration thread",
                thread_index=self.thread_index,
                worker_index=self.worker.index(),
                shuffle=self.shuffle,
                class_name="OrchestrateSingleWorker",
                class_method="run_without_buffer",
            )
            while True:
                if self.close_event.is_set():
                    # simply return if close event
                    self.parent.debug(
                        "Received close event from parent. Terminating thread now",
                        thread_index=self.thread_index,
                        worker_index=self.worker.index(),
                        shuffle=self.shuffle,
                        class_name="OrchestrateSingleWorker",
                        class_method="run_without_buffer",
                    )
                    return

                # parent will unset epoch end event
                # so when it is unset, then we can pull
                # the idea is that we need to wait for all
                # workers to finish their epoch before starting new ones
                if not self.epoch_end.is_set():
                    data = self.worker.get()

                    if data is CHILD_MESSAGE.EPOCH_END:
                        # set epoch end event to let parent knows
                        self.epoch_end.set()
                        # still need to put epoch end signal into queue
                        if self.shuffle:
                            # if empty, then try to put to the queue
                            while not self.data_queue.put(CHILD_MESSAGE.EPOCH_END):
                                time.sleep(0.0001)
                                # need to check close event to avoid deadlock
                                if self.close_event.is_set():
                                    # simply return if close event
                                    return
                        else:
                            # note that if not shuffle, we are using OrderDataQueue
                            # EPOCH_END bypasses turn ordering, should succeed immediately
                            if self.close_event.is_set():
                                return
                            success = self.data_queue.put(CHILD_MESSAGE.EPOCH_END, self.thread_index)
                            if not success:
                                # Should not happen for EPOCH_END
                                time.sleep(0.001)

                        self.parent.debug(
                            "Received epoch end signal in orchestration thread",
                            thread_index=self.thread_index,
                            worker_index=self.worker.index(),
                            shuffle=self.shuffle,
                            class_name="OrchestrateSingleWorker",
                            class_method="run_without_buffer",
                        )
                    elif data is not None:
                        if self.shuffle:
                            # put until successful
                            while not self.data_queue.put(data):
                                time.sleep(0.0001)
                                # need to check close event to avoid deadlock
                                if self.close_event.is_set():
                                    return

                        else:
                            # put into queue
                            # Note: put() now blocks internally when shuffle=False, so no need for busy-waiting
                            if self.shuffle:
                                # For shuffle=True, put() may still need retries if queue is full
                                while not self.data_queue.put(data):
                                    time.sleep(0.0001)
                                    if self.close_event.is_set():
                                        return
                            else:
                                # For shuffle=False, put() returns False if not our turn or queue full
                                # Use condition variable to wait efficiently
                                success = self.data_queue.put(data, self.thread_index)
                                if not success:
                                    # Not our turn or queue full - wait on condition
                                    with self.data_queue.condition:
                                        if self.close_event.is_set():
                                            return
                                        # Wait with timeout to periodically check close_event
                                        self.data_queue.condition.wait(timeout=0.1)
                                    # After wait, try again in next iteration
                else:
                    time.sleep(0.001)

        except BaseException as error:
            # get traceback content
            traceback_content = traceback.format_exc()
            # log and print
            self.parent.error(
                str(error),
                thread_index=self.thread_index,
                worker_index=self.worker.index(),
                shuffle=self.shuffle,
                class_name="OrchestrateSingleWorker",
                class_method="run_without_buffer",
                traceback=traceback_content,
            )
            print(traceback_content)

            # set event so parent knows
            self.close_event.set()

            # then raise error
            raise error

    def run(self):
        if self.use_buffer:
            self.run_with_buffer()
        else:
            self.run_without_buffer()


def orchestrate_multiple_workers(
    shuffle: bool,
    seed: int,
    workers: list[Worker],
    data_queue: Queue,
    max_queue_size: int,
    close_event: threading.Event,
    epoch_end_events: list[threading.Event],
    parent: WorkerManager,
):
    """
    Thread to pull data from many workers and put into data queue, which
    is read by the iterator in WorkerManager
    """

    random.seed(seed)
    try:
        # indices of active workers
        indices = list(range(len(workers)))
        while True:
            if close_event.is_set():
                # simply return if close event
                return

            removed_indices = []
            for idx in indices:
                if shuffle:
                    # if shuffling is enabled, we can pull without waiting
                    data = workers[idx].get()
                    if data is CHILD_MESSAGE.EPOCH_END:
                        # set epoch end event to let parent knows
                        epoch_end_events[idx].set()
                        removed_indices.append(idx)

                        # still need to put epoch end signal into queue
                        # epoch end will be forcefully put so it's always successful
                        data_queue.put(CHILD_MESSAGE.EPOCH_END)
                        parent.debug(
                            "Receved epoch end signal in orchestration thread",
                            method="orchestrate_multiple_workers",
                        )

                    elif data is not None:
                        # put into queue
                        # note that if queue is full, then need to wait
                        # put returns True if putting data successfully
                        while not data_queue.put(data):
                            time.sleep(0.005)
                            # remember to check close event to avoid deadlock
                            if close_event.is_set():
                                # simply return if close event
                                return
                else:
                    # need to get() until having data or epoch end signal from worker
                    # is received
                    while True:
                        data = workers[idx].get()
                        if data is CHILD_MESSAGE.EPOCH_END:
                            # set epoch end event to let parent knows
                            epoch_end_events[idx].set()
                            removed_indices.append(idx)

                            # still need to put epoch end signal into queue
                            # epoch end will be forcefully put so it's always successful
                            data_queue.put(CHILD_MESSAGE.EPOCH_END)
                            break
                        elif data is not None:
                            while not data_queue.put(data):
                                time.sleep(0.005)
                                if close_event.is_set():
                                    # simply return if close event
                                    return
                            break
                        else:
                            time.sleep(0.001)

                        # remember to check close event to avoid deadlock
                        if close_event.is_set():
                            # simply return if close event
                            return

            for idx in removed_indices:
                indices.remove(idx)

            # now if indices is empty, it means epoch has ended
            # we need check if all epoch ended flags have been unset
            if len(indices) == 0:
                flag_unset = 0
                for event in epoch_end_events:
                    if not event.is_set():
                        flag_unset += 1

                if flag_unset == len(workers):
                    # reset indices
                    indices = list(range(len(workers)))
                    if shuffle:
                        random.shuffle(indices)

    except BaseException as error:
        # get traceback content
        traceback_content = traceback.format_exc()
        # log and print
        parent.error(
            str(error),
            method="orchestrate_multiple_workers",
            traceback=traceback_content,
        )
        print(traceback_content)

        # set event so parent knows
        close_event.set()

        # then raise error
        raise error


class DataQueue:
    """
    Data queue used in parent process to store reconstructed data
    This queue handles shuffled data
    """

    def __init__(
        self,
        size: int,
        device: Any,
        to_device: Callable,
        benchmark: bool,
        benchmark_logger: Any,
        max_nb_batch_on_device: int,
    ):
        self.max_size = size
        self.device = device
        if device is not None and to_device is None:
            self.to_device = get_default_to_device()
        else:
            self.to_device = to_device

        self.data = []
        self.lock = threading.Lock()
        self.nb_batch_on_device = 0
        self.max_nb_batch_on_device = max_nb_batch_on_device

        if benchmark:
            self.injection = ThroughputMeasure(
                max_count=size - 1, task_name="injection", logger=benchmark_logger
            )
            self.consumption = ThroughputMeasure(
                max_count=size - 1, task_name="consumption", logger=benchmark_logger
            )
            self.transfer_to_device = BenchmarkProperty(max_count=size - 1)
            self.time_measure = TimeMeasure
        else:
            self.injection = None
            self.consumption = None
            self.transfer_to_device = None
            self.time_measure = DummyTimeMeasure

        self.benchmark = benchmark
        self.logger = benchmark_logger

    def put(self, item: Any, dummy: int):
        with self.lock:
            self.injection.start()

            if item is CHILD_MESSAGE.EPOCH_END:
                # if epoch end, forcefully put and return True
                self.data.append(CHILD_MESSAGE.EPOCH_END)

                return True

            if len(self.data) <= self.max_size:
                self.data.append(item)
                success = True
                # increase counter for throughput measurement
                self.injection.count()
            else:
                success = False

            # move top item to device if needed
            if (
                self.device is not None
                and self.nb_batch_on_device < len(self.data)
                and self.nb_batch_on_device < self.max_nb_batch_on_device
                and self.data[self.nb_batch_on_device] is not CHILD_MESSAGE.EPOCH_END
            ):
                with self.time_measure(
                    self.transfer_to_device, self.logger, "transfer to device"
                ):
                    self.data[self.nb_batch_on_device] = self.to_device(
                        self.data[self.nb_batch_on_device], self.device
                    )
                self.nb_batch_on_device += 1

            return success

    def get(self):
        with self.lock:
            if len(self.data) == 0:
                return

            # throughput measurement will mark the time
            self.consumption.start()

            item = self.data.pop(0)

            if (
                self.device is not None
                and self.nb_batch_on_device == 0
                and item is not CHILD_MESSAGE.EPOCH_END
            ):
                # note that if number of batches on device is 0, meaning
                # it has not been moved to device
                with self.time_measure(
                    self.transfer_to_device, self.logger, "transfer to device"
                ):
                    item = self.to_device(item, self.device)
                    self.nb_batch_on_device += 1

            if item is not CHILD_MESSAGE.EPOCH_END:
                self.nb_batch_on_device -= 1

            self.consumption.count()
            return item


class OrderDataQueue:
    """
    Data queue used in parent process to store reconstructed data
    This queue handles non-shuffled data, i.e., a batch is put
    into the queue only when its turn comes
    """

    def __init__(
        self,
        size: int,
        nb_thread: int,
        device: Any,
        to_device: Callable,
        benchmark: bool,
        benchmark_logger: Any,
        max_nb_batch_on_device: int,
    ):
        self.max_size = size
        self.device = device
        if device is not None and to_device is None:
            self.to_device = get_default_to_device()
        else:
            self.to_device = to_device

        self.data = []
        self.lock = threading.Lock()
        # Use Condition for proper blocking instead of busy-waiting
        self.condition = threading.Condition(self.lock)
        self.top_on_device = False
        self.cur_thread_idx = 0
        self.all_thread_indices = list(range(nb_thread))
        self.nb_thread = nb_thread
        self.max_nb_batch_on_device = max_nb_batch_on_device
        self.nb_batch_on_device = 0

        if benchmark:
            self.injection = ThroughputMeasure(
                max_count=size - 1, task_name="injection", logger=benchmark_logger
            )
            self.consumption = ThroughputMeasure(
                max_count=size - 1, task_name="consumption", logger=benchmark_logger
            )
            self.transfer_to_device = BenchmarkProperty(max_count=size - 1)
            self.time_measure = TimeMeasure
        else:
            self.injection = None
            self.consumption = None
            self.transfer_to_device = None
            self.time_measure = DummyTimeMeasure
        self.benchmark = benchmark
        self.logger = benchmark_logger

    def update_indices(self, epoch_end=False):
        if epoch_end:
            if self.cur_thread_idx == len(self.all_thread_indices) - 1:
                # if last index --> next index will be 0
                self.all_thread_indices.pop(self.cur_thread_idx)
                self.cur_thread_idx = 0
            else:
                # not last, then we just keep the current index without incrementing
                # because we already remove the cur element
                self.all_thread_indices.pop(self.cur_thread_idx)

            if len(self.all_thread_indices) == 0:
                # if no more thread, then reset
                self.all_thread_indices = list(range(self.nb_thread))
        else:
            self.cur_thread_idx = (self.cur_thread_idx + 1) % len(
                self.all_thread_indices
            )
        # Notify waiting threads that the turn has changed
        self.condition.notify_all()

    def put(self, item: Any, thread_index: int):
        with self.lock:
            self.injection.start()
            
            # EPOCH_END signals should bypass turn ordering to prevent deadlocks
            # They can be put immediately regardless of turn
            if item is CHILD_MESSAGE.EPOCH_END:
                self.data.append(CHILD_MESSAGE.EPOCH_END)
                self.update_indices(epoch_end=True)
                # Notify waiting threads that turn may have changed
                self.condition.notify_all()
                return True
            
            # For regular items, check if it's this thread's turn
            if self.cur_thread_idx != thread_index:
                # Not our turn yet
                return False

            # Check if queue is full
            if len(self.data) > self.max_size:
                # Queue is full
                return False

            # Now we can put the item
            self.data.append(item)
            self.update_indices()
            self.injection.count()
            success = True

            # move top item to device if needed
            if (
                self.device is not None
                and self.nb_batch_on_device < len(self.data)
                and self.nb_batch_on_device < self.max_nb_batch_on_device
                and self.data[self.nb_batch_on_device] is not CHILD_MESSAGE.EPOCH_END
            ):
                with self.time_measure(
                    self.transfer_to_device, self.logger, "transfer to device"
                ):
                    self.data[self.nb_batch_on_device] = self.to_device(
                        self.data[self.nb_batch_on_device], self.device
                    )
                self.nb_batch_on_device += 1

            # Notify waiting threads that turn changed and queue space is available
            self.condition.notify_all()

            return success

    def get(self):
        with self.lock:
            if len(self.data) == 0:
                return

            # throughput measurement will mark the time
            self.consumption.start()

            item = self.data.pop(0)

            if (
                self.device is not None
                and self.nb_batch_on_device == 0
                and item is not CHILD_MESSAGE.EPOCH_END
            ):
                # note that if number of batches on device is 0, meaning
                # it has not been moved to device
                with self.time_measure(
                    self.transfer_to_device, self.logger, "transfer to device"
                ):
                    item = self.to_device(item, self.device)
                    self.nb_batch_on_device += 1

            if item is not CHILD_MESSAGE.EPOCH_END:
                self.nb_batch_on_device -= 1

            self.consumption.count()
            
            # Notify waiting threads that queue space is now available
            self.condition.notify_all()
            
            return item


class Worker(CTX.Process):
    """
    Worker process that loads data and batches them
    """

    def __init__(
        self,
        dataset_file: str,
        nb_consumer: int,
        worker_per_consumer: int,
        consumer_index: int,
        worker_index: int,
    ):
        super().__init__()

        # create pipes for telcom
        self.front_read_pipe, self.back_write_pipe = CTX.Pipe()
        self.back_read_pipe, self.front_write_pipe = CTX.Pipe()
        self.dataset_file = dataset_file
        self.nb_consumer = nb_consumer
        self.worker_per_consumer = worker_per_consumer
        self.consumer_index = consumer_index
        self.worker_index = worker_index
        self.name = f"Worker-{worker_index}"
        self.side = "parent"

        # worker_states are only visible to parent
        self.worker_states = Property()
        self.worker_states.started = False
        self.worker_states.closed = False
        self.worker_states.ready = False
        self.worker_states.nb_batch = None
        self.worker_states.child_got_noticed = False
        self.logger = None
        self.shared_memory = None

    def index(self) -> int:
        """
        Return worker index
        """
        return self.worker_index

    def start(self):
        # note that start() should not be decorated with child_task_guard
        if not self.worker_states.started:
            # calling superclass
            super().start()

            # logger
            with open(self.dataset_file, "rb") as f:
                kwargs = dill.load(f)

            self.logger = self.get_logger(**kwargs)

            # decoder
            self.batch_decoder = kwargs["batch_decoder"]

            # mark as started
            self.worker_states.started = True
            self.debug("Worker started", class_method="start")

    def wait_until_ready(self):
        """
        Wait until receiving the number of minibatch is sent from child
        Note: this should not be decorated with child_task_guard
        because this method is called in parent side
        """

        if not self.worker_states.started:
            self.error(
                f"Worker ({self.name}) has not been started",
                class_method="wait_until_ready",
            )
            raise RuntimeError(f"Worker ({self.name}) has not been started")

        while True:
            if self.front_read_pipe.poll():
                resp_code = self.front_read_pipe.recv()

                # check if this is metadata
                if not isinstance(resp_code, dict):
                    self.error(
                        f"Metadata is expected to be a dict. Unexpected response from child: {resp_code}",
                        class_method="wait_until_ready",
                    )
                    raise ValueError(
                        f"Metadata is expected to be a dict. Unexpected response from child: {resp_code}"
                    )

                # check response
                resp_code["type"] = CHILD_MESSAGE.decode(resp_code["type"])
                if resp_code["type"] == CHILD_MESSAGE.METADATA:
                    # save number of minibatches in one epoch
                    self.worker_states.nb_batch = resp_code["nb_batch"]

                    self.worker_states.batch_count = 0

                    # shared memory to read data
                    self.shared_memory = multiprocessing.shared_memory.SharedMemory(
                        name=resp_code["shared_memory"], create=False
                    )
                    self.debug(
                        "Receive metadata from child",
                        class_method="start",
                        nb_batch=self.worker_states.nb_batch,
                        shared_memory_name=resp_code["shared_memory"],
                    )

                    # need to send ack to child
                    self.front_write_pipe.send(PARENT_MESSAGE.ACK.value)

                    self.worker_states.ready = True
                    return
                else:
                    self.error(
                        "Metadata is expected from 1st message from child process",
                        class_method="wait_until_ready",
                        received=resp_code,
                    )
                    raise ValueError(
                        "Metadata is expected from 1st message from child process"
                    )
            else:
                time.sleep(0.01)

    def __len__(self):
        if not self.worker_states.ready:
            self.error(
                "Worker is not ready yet. Please call wait_until_ready() first",
                class_method="__len__",
            )
            raise RuntimeError(
                f"Worker ({self.name}) is not ready yet. Please call wait_until_ready() first"
            )
        return self.worker_states.nb_batch

    def notify_child(self):
        """
        Tell child to close
        """
        if not self.worker_states.child_got_noticed:
            self.front_write_pipe.send(CHILD_MESSAGE.TERMINATE.value)
            self.worker_states.child_got_noticed = True

    def close(self):
        """
        Clean up resources
        """

        if not self.worker_states.closed:
            # tell child to close
            self.notify_child()

            # close shared memory
            if self.shared_memory is not None:
                # we dont unlink because child owns it
                self.shared_memory.close()

            # close pipes
            self.front_write_pipe.close()

            # wait until child exits
            self.debug(
                "Waiting for child process to join",
                class_method="close",
            )
            self.join()
            self.worker_states.closed = True
            self.debug("Child process has joined", class_method="close")

    @parent_task_guard
    def get(self):
        """
        This function attempt to load data from queue
        Return None immediately if queue is empty
        """
        # double check
        if not self.worker_states.ready:
            self.error(
                "Worker is not ready yet. Please call wait_until_ready() first",
                class_method="get",
            )
            raise RuntimeError(
                f"Worker ({self.name}) is not ready yet. Please call wait_until_ready() first"
            )

        if self.front_read_pipe.poll():
            # if has something in the queue
            resp = self.front_read_pipe.recv()
            if isinstance(resp, tuple):
                # get start and stop indices in the shared memory
                start, stop = resp

                # read bytes
                self.debug(
                    f"Got start={start}, stop={stop} index of shared_memory from child",
                    class_method="get",
                )
                batch_in_bytes = bytes(self.shared_memory.buf[start:stop])

                # send ack to child
                self.front_write_pipe.send(PARENT_MESSAGE.ACK.value)

                # decode
                batch = self.batch_decoder(batch_in_bytes)

                self.worker_states.batch_count += 1

                return batch

            if isinstance(resp, dict):
                # this is error from child
                self.error(
                    "Error happens in child process",
                    class_method="get",
                    error=resp["error"],
                    traceback=resp["traceback"],
                )
                print(resp["traceback"])

                raise RuntimeError(f"{self.name} receives error from child: {resp}")

            elif isinstance(resp, int):
                resp = CHILD_MESSAGE.decode(resp)

                if resp == CHILD_MESSAGE.EPOCH_END:
                    # Note: nb_batch may change after rotation, so we validate that we received
                    # at least one batch (batch_count > 0) rather than checking exact modulo
                    # The child process ensures the correct number of batches is sent
                    if self.worker_states.batch_count == 0:
                        self.error(
                            "Receive epoch end signal from child but no batches were received",
                            class_method="get",
                        )
                        raise RuntimeError(
                            "Receive epoch end signal from child but no batches were received"
                        )

                    # Reset batch count for next epoch (nb_batch may change after rotation)
                    batches_received = self.worker_states.batch_count
                    self.worker_states.batch_count = 0

                    # send ack to child
                    self.front_write_pipe.send(PARENT_MESSAGE.ACK.value)
                    self.debug(
                        f"Receive epoch end signal from child, received {batches_received} batches",
                        class_method="get"
                    )

                    # return epoch end signal to any processor here
                    # to know this is the end of epoch
                    return CHILD_MESSAGE.EPOCH_END
                else:
                    self.error(
                        f"Unexpected response from child: {resp}",
                        class_method="get",
                    )
                    raise ValueError(f"Unexpected response from child: {resp}")

    @child_task_guard
    def prepare_dataset(self, **kwargs):
        # load dataset
        constructor = kwargs["dataset_class"]
        params = kwargs["dataset_kwargs"]
        self.data.seed = kwargs["seed"]
        self.data.dataset = constructor(**params)

        # compute the indices of the samples to be loaded
        # by this worker
        nb_worker = self.nb_consumer * self.worker_per_consumer
        self.data.shuffle = kwargs["shuffle"]
        self.data.bs = kwargs["batch_size"]
        self.data.nearby_shuffle = kwargs["nearby_shuffle"]

        # find relevant sample indices
        if self.data.shuffle:
            # if shuffle, then dataset is divided into K (nb_worker) segments
            # each segment has consecutive samples that a worker will process
            # store nb_worker and worker_size for rotation in later epochs
            self.data.nb_worker = nb_worker
            self.data.worker_size = int(np.ceil(len(self.data.dataset) / nb_worker))
            # start, stop sample index (for epoch 0, use original worker_index)
            # rotation will happen in collect_minibatch when epoch ends
            self.data.start = self.worker_index * self.data.worker_size
            self.data.stop = min(self.data.start + self.data.worker_size, len(self.data.dataset))
            self.data.indices = list(range(self.data.start, self.data.stop))
            self.debug(
                f"Start sample index={self.data.start}, stop sample index={self.data.stop}",
                class_method="prepare_dataset",
            )

        else:
            # if not shuffle, the dataset is divided into N segments
            # N is the number of consumers
            # the n-th segment belongs to n-th consumer
            # this segment is divided into minibatches
            # the i-th minibatch is processed by (i % worker_per_consumer) worker
            self.data.indices = []
            # first, we need to find the start and stop of the consumer segment
            consumer_size = int(np.ceil(len(self.data.dataset) / self.nb_consumer))
            # start and stop are sample indices of the consumer
            consumer_start = self.consumer_index * consumer_size
            consumer_stop = min(consumer_start + consumer_size, len(self.data.dataset))
            self.debug(
                f"Consumer start sample index={consumer_start}, stop index={consumer_stop}",
                class_method="prepare_dataset",
            )
            # find total number of batches processed by this consumer
            consumer_batch_count = int(
                np.ceil((consumer_stop - consumer_start) / self.data.bs)
            )
            # find the local index of the worker within the consumer
            local_worker_index = self.worker_index % self.worker_per_consumer

            for i in range(consumer_batch_count):
                if (i % self.worker_per_consumer) == local_worker_index:
                    start = consumer_start + i * self.data.bs
                    stop = min(start + self.data.bs, consumer_stop)
                    self.data.indices.extend(list(range(start, stop)))

        # current sample index
        self.data.cur = 0
        # current epoch
        self.data.epoch = 0
        # number of minibatches handled by this worker
        self.data.nb_batch = int(np.ceil(len(self.data.indices) / self.data.bs))

        # handle collate
        if kwargs["collate_fn"] is None:
            self.data.collate_fn = get_default_collate_fn()
        else:
            self.data.collate_fn = kwargs["collate_fn"]

        # handle encoder, decoder
        if kwargs["batch_encoder"] is None:
            self.data.encoder = dill.dumps
        else:
            self.data.encoder = kwargs["batch_encoder"]

        # prepare benchmark
        self.benchmark = Property()
        self.benchmark.enable = kwargs["benchmark"]
        self.benchmark.collect = BenchmarkProperty(max_count=kwargs["data_queue_size"])
        self.benchmark.collate = BenchmarkProperty(max_count=kwargs["data_queue_size"])
        self.benchmark.encode = BenchmarkProperty(max_count=kwargs["data_queue_size"])

        # prepare shared memory
        # try to compute the size of a minibatch first
        bs_size = len(
            self.data.encoder(
                self.data.collate_fn(
                    [self.data.dataset[i] for i in self.data.indices[: self.data.bs]]
                )
            )
        )
        # allocate shared memory
        self.data.mem_queue = SharedMemoryQueue(
            size=bs_size * kwargs["data_queue_size"],
            size_in_bytes=bs_size * kwargs["data_queue_size"],
            logger=self.logger,
            benchmark=kwargs["benchmark"],
            benchmark_logger=self.benchmark_logger,
        )

    def collect_minibatch(self):
        stop = min(self.data.cur + self.data.bs, len(self.data.indices))
        data = [self.data.dataset[i] for i in self.data.indices[self.data.cur : stop]]

        if stop == len(self.data.indices):
            # end of epoch
            self.data.cur = 0

            # reshuffling
            if self.data.shuffle:
                self.data.seed += 1
                # rotate the worker index for the next epoch
                # rotated_worker_index = (worker_index + epoch + 1) % nb_worker
                # Note: epoch will be incremented after this, so we use epoch + 1
                rotated_worker_index = (self.worker_index + self.data.epoch + 1) % self.data.nb_worker
                # recalculate start and stop based on rotated worker index
                self.data.start = rotated_worker_index * self.data.worker_size
                self.data.stop = min(self.data.start + self.data.worker_size, len(self.data.dataset))
                # perform nearby shuffle to avoid too random reading
                self.data.indices = shuffle_indices(
                    self.seed, self.data.start, self.data.stop, self.data.nearby_shuffle
                )
                # update nb_batch in case the number of samples changed after rotation
                self.data.nb_batch = int(np.ceil(len(self.data.indices) / self.data.bs))
                self.debug(
                    f"Epoch {self.data.epoch + 1}: Rotated to worker_index={rotated_worker_index}, "
                    f"start={self.data.start}, stop={self.data.stop}, nb_batch={self.data.nb_batch}",
                    class_method="collect_minibatch",
                )
            self.data.epoch += 1

            epoch_time = time.perf_counter() - self.data.epoch_start
            self.data.epoch_start = time.perf_counter()
            self.debug(
                f"Epoch {self.data.epoch} took {epoch_time:.2f} seconds",
                class_method="collect_minibatch",
            )
        else:
            self.data.cur = stop
        return data

    def get_logger(self, **kwargs):
        if kwargs["logger"] is None:
            logger = None
        else:
            logger = kwargs["logger"].copy()
            logger["name"] = f"{self.name}|{self.side}"
            logger["suffix"] = f"{self.name}|{self.side}"

        return get_logger(**logger)

    @child_task_guard
    def start_signal_handler(self, message_queue_size: int):
        # start a thread to watch for close event
        # Pass self (Worker instance) instead of logger so signal_handler can access self.data.nb_batch dynamically
        self.signal_thread = threading.Thread(
            target=signal_handler,
            args=(
                self.data.nb_batch,  # initial value, but signal_handler will read current value dynamically
                self.back_read_pipe,
                self.back_write_pipe,
                self.data.mem_queue,
                self.close_event,
                self,  # Pass Worker instance so signal_handler can access self.data.nb_batch
                message_queue_size,
            ),
        )
        self.signal_thread.start()

    def notify_parent(self, **message: dict):
        self.back_write_pipe.send(message)

    def extend_clarifications(self, **clarifications):
        clarifications["name"] = self.name
        clarifications["class_name"] = self.__class__.__name__
        clarifications["side"] = self.side
        return clarifications

    def print_log(self, message, **clarifications):
        clarifications = [f"{key}={value}" for key, value in clarifications.items()]
        clarifications = ", ".join(clarifications)
        print(f"{message} ({clarifications})")

    def debug(self, message, **clarifications):
        clarifications = self.extend_clarifications(**clarifications)
        if self.logger is None:
            self.print_log(message, **clarifications)
            return

        self.logger.debug(
            message,
            **clarifications,
        )

    def info(self, message, **clarifications):
        clarifications = self.extend_clarifications(**clarifications)
        if self.logger is None:
            self.print_log(message, **clarifications)
            return

        self.logger.info(
            message,
            **clarifications,
        )

    def warning(self, message, **clarifications):
        clarifications = self.extend_clarifications(**clarifications)
        if self.logger is None:
            self.print_log(message, **clarifications)
            return

        self.logger.warning(
            message,
            **clarifications,
        )

    def error(self, message: str, **clarifications: dict):
        clarifications = self.extend_clarifications(**clarifications)
        if self.logger is None:
            self.print_log(message, **clarifications)
            return

        self.logger.warning(
            message,
            **clarifications,
        )

    def clean_child(self):
        self.debug(
            "Start cleaning child process",
            class_method="clean_child",
        )

        if not self.is_child_clean:
            # if signal thread has been launched
            if self.signal_thread is not None:
                self.debug(
                    "Signal thread has been launched. Waiting for it to join...",
                    class_method="clean_child",
                )
                # set close event
                self.close_event.set()
                # wait until it joins before cleaning other resources
                self.signal_thread.join()
                self.debug("Signal thread has joined", class_method="clean_child")

            # clean shared memory
            if hasattr(self.data, "mem_queue"):
                self.data.mem_queue.close()
                self.debug(
                    f"Shared memory queue {self.data.mem_queue.name()} has been cleaned",
                    class_method="clean_child",
                )

            # close back pipe
            self.back_write_pipe.close()
            self.is_child_clean = True

    @child_task_guard
    def init_child(self, **kwargs):
        self.is_child_clean = False
        self.signal_thread = None

        # namespace for data
        self.data = Property()
        # get logger
        self.logger = self.get_logger(**kwargs)

        # if benchmark is enable, prepare tools to benchmark
        if kwargs["benchmark"]:
            # tool to measure time
            self.time_measure = TimeMeasure

            # if a separate sink was set, use it
            if kwargs["benchmark_file"] is not None:
                config = kwargs["logger"].copy()
                config["name"] = f"{self.name}"
                config["suffix"] = None
                config["path"] = kwargs["benchmark_file"]
                self.benchmark_logger = get_logger(**config)
            else:
                self.benchmark_logger = self.logger
        else:
            self.time_measure = DummyTimeMeasure
            self.benchmark_logger = None

        self.seed = kwargs["seed"]

    @child_task_guard
    def generate_data(self):
        # start main loop
        cur_batch = None

        # start time
        self.data.epoch_start = time.perf_counter()

        while True:
            # collect data for minibatch
            if cur_batch is None:
                with self.time_measure(
                    self.benchmark.collect,
                    self.benchmark_logger,
                    task_name="collect batch data",
                ):
                    cur_batch = self.collect_minibatch()

                # collate data
                with self.time_measure(
                    self.benchmark.collate,
                    self.benchmark_logger,
                    task_name="collate batch data",
                ):
                    cur_batch = self.data.collate_fn(cur_batch)

                # serialize
                with self.time_measure(
                    self.benchmark.encode,
                    self.benchmark_logger,
                    task_name="serialize batch data",
                ):
                    cur_batch = self.data.encoder(cur_batch)

            # put into queue
            # note that putting successfully should return True, otherwise False
            # we need to wait until this is put successfully
            success = self.data.mem_queue.put(cur_batch)
            if success:
                cur_batch = None
            else:
                time.sleep(0.001)

            # check close signal
            if self.close_event.is_set():
                # this comes from signal thread, which comes from parent
                # or error in signal thread
                # we just need to clean up the child and exit without
                # notifying parent
                self.clean_child()
                return

    def run(self):
        self.side = "child"
        # load kwargs
        with open(self.dataset_file, "rb") as f:
            kwargs = dill.load(f)

        # init
        self.init_child(**kwargs)

        # flag to check terminate signal
        self.close_event = threading.Event()

        # prepare dataset
        self.prepare_dataset(**kwargs)

        # start signal handler
        self.start_signal_handler(kwargs["message_queue_size"])

        # start generating data
        self.generate_data()


class WorkerManager:
    """
    Manager that handles multiple workers
    Also expose iterator interface
    """

    REQUIRED_PARAMS = [
        "dataset_class",
        "dataset_kwargs",
        "shuffle",
        "nb_worker",
        "batch_size",
        "nearby_shuffle",
        "collate_fn",
        "batch_encoder",
        "batch_decoder",
        "benchmark",
        "benchmark_file",
        "data_queue_size",
        "message_queue_size",
        "logger",
        "seed",
    ]

    def __init__(
        self,
        dataset_file: str,
        nb_consumer: int,
        worker_per_consumer: int,
        shuffle: bool,
        seed: int,
        multithread: bool,
        data_queue_size: int,
        max_nb_batch_on_device: int,
        max_buffer_size: int,
    ):
        self.verify_params(dataset_file)
        self.dataset_file = dataset_file
        self.nb_consumer = nb_consumer
        self.worker_per_consumer = worker_per_consumer
        self.device = None
        self.to_device = None
        self.shuffle = shuffle
        self.seed = seed
        self.data_queue_size = data_queue_size
        self.max_buffer_size = max_buffer_size
        self.max_nb_batch_on_device = max_nb_batch_on_device
        self.multithread = multithread
        self.orchestrate_thread = None
        self.worker = Property()
        self.worker.processes = []
        self.worker.threads = []
        self.worker.close_event = None
        self.worker.epoch_end = None
        self.started = False
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __len__(self):
        if not self.started:
            self.error("start() should be called first", class_method="__len__")
            raise RuntimeError("start() should be called first")

        return self.nb_batch

    def __iter__(self):
        self.nb_finished_worker = 0
        self.batch_count = 0
        return self

    def __next__(self):
        epoch_ended = False
        try:
            while not epoch_ended:
                # get item from data queue
                item = self.worker.data_queue.get()

                if item is None:
                    # if nothing, sleep a bit then skip to next iter
                    time.sleep(0.001)
                    continue

                if item is CHILD_MESSAGE.EPOCH_END:
                    # if epoch end,
                    self.nb_finished_worker += 1
                    self.debug(
                        "Receive epoch ended signal from child",
                        class_method="__next__",
                        nb_finished_worker=self.nb_finished_worker,
                    )

                    # if all workers have finished, need to reset the flags
                    if self.nb_finished_worker == self.worker_per_consumer:
                        self.nb_finished_worker = 0
                        for event in self.worker.epoch_end:
                            event.clear()

                        epoch_ended = True
                        self.debug("Epoch has now ended", class_method="__next__")
                    continue
                break

        except BaseException as error:
            traceback_content = traceback.format_exc()
            self.error(str(error), class_method="__next__", traceback=traceback_content)
            self.close()
            raise error

        if not epoch_ended:
            if item is None:
                self.error(
                    "Data queue is empty but epoch has not ended",
                    class_method="__next__",
                )
                raise RuntimeError("Data queue is empty but epoch has not ended")

            self.batch_count += 1

            if self.device is not None:
                item = self.to_device(item, self.device)
            return item
        else:
            raise StopIteration

    def verify_params(self, dataset_file: dict):
        with open(dataset_file, "rb") as f:
            params = dill.load(f)

        for key in self.REQUIRED_PARAMS:
            if key not in params:
                raise ValueError(f"Missing required parameter {key}")

    def start(
        self,
        consumer_index: int,
        device: Any = None,
        to_device: Callable = None,
    ):
        """
        Start the workers for a given consumer index
        """

        if self.started:
            raise RuntimeError(
                "Worker manager has been started. Should only be called once"
            )

        self.consumer_index = consumer_index
        self.device = device

        if to_device is None:
            self.to_device = get_default_to_device()
        else:
            self.to_device = to_device

        self.get_logger()

        if consumer_index >= self.nb_consumer:
            self.error(
                "Consumer index >= number of consumer",
                consumer_index=consumer_index,
                nb_consumer=self.nb_consumer,
            )
            raise ValueError(
                f"Consumer index ({consumer_index}) >= number of consumer ({self.nb_consumer})"
            )

        # start workers
        # start workers
        self.start_workers()

        # then wait until ready
        self.wait_until_ready()

        # then start thread to handle workers
        self.worker.close_event = threading.Event()

        self.start_worker_threads()
        self.started = True

    def close(self):
        if not self.closed:
            # set close event, which is monitored by threads
            if self.worker.close_event is not None:
                self.debug("Setting close event", class_method="close")
                self.worker.close_event.set()
                time.sleep(1)

            # tell children
            self.debug("Telling child processes to close", class_method="close")
            for worker in self.worker.processes:
                worker.notify_child()

            # then close
            self.debug(
                "Closing child processes and waiting for them to join",
                class_method="close",
            )
            for worker in self.worker.processes:
                worker.close()

            # tell orchestrate threads to close
            self.debug("Waiting for orchestrate threads to join", class_method="close")
            for thread in self.worker.threads:
                thread.join()
            self.debug("Orchestrate threads have joined", class_method="close")

            self.closed = True

    @manager_task_guard
    def start_workers(self):
        for i in range(self.worker_per_consumer):
            worker = Worker(
                dataset_file=self.dataset_file,
                nb_consumer=self.nb_consumer,
                worker_per_consumer=self.worker_per_consumer,
                consumer_index=self.consumer_index,
                worker_index=self.consumer_index * self.worker_per_consumer + i,
            )
            worker.start()
            self.worker.processes.append(worker)

    @manager_task_guard
    def wait_until_ready(self):
        for worker in self.worker.processes:
            worker.wait_until_ready()

        self.nb_batch = sum([len(worker) for worker in self.worker.processes])

    @manager_task_guard
    def start_worker_threads(self):
        if self.shuffle or not self.multithread:
            self.worker.data_queue = DataQueue(
                size=self.data_queue_size,
                device=self.device,
                to_device=self.to_device,
                benchmark=self.benchmark,
                benchmark_logger=self.benchmark_logger,
                max_nb_batch_on_device=self.max_nb_batch_on_device,
            )
        else:
            self.worker.data_queue = OrderDataQueue(
                size=self.data_queue_size,
                nb_thread=self.worker_per_consumer,
                device=self.device,
                to_device=self.to_device,
                benchmark=self.benchmark,
                benchmark_logger=self.benchmark_logger,
                max_nb_batch_on_device=self.max_nb_batch_on_device,
            )

        if self.multithread:
            # if using 1 thread for 1 worker
            self.worker.epoch_end = []
            for wrk_idx, worker in enumerate(self.worker.processes):
                self.worker.epoch_end.append(threading.Event())
                thread = OrchestrateSingleWorker(
                    shuffle=self.shuffle,
                    thread_index=wrk_idx,
                    worker=worker,
                    data_queue=self.worker.data_queue,
                    close_event=self.worker.close_event,
                    epoch_end=self.worker.epoch_end[-1],
                    parent=self,
                    max_buffer_size=self.max_buffer_size,
                )
                thread.start()
                self.worker.threads.append(thread)
        else:
            # using a single thread to manage all workers

            self.worker.epoch_end = [threading.Event() for _ in self.worker.processes]
            self.worker.threads.append(
                threading.Thread(
                    target=orchestrate_multiple_workers,
                    args=(
                        self.shuffle,
                        self.seed,
                        self.worker.processes,
                        self.worker.data_queue,
                        self.max_buffer_size,
                        self.worker.close_event,
                        self.worker.epoch_end,
                        self,
                    ),
                )
            )
            self.worker.threads[-1].start()

    def get_logger(self):
        with open(self.dataset_file, "rb") as fid:
            kwargs = dill.load(fid)

        if kwargs["logger"] is None:
            logger = None
        else:
            logger = kwargs["logger"].copy()
            logger["name"] = f"WorkerManager-{self.consumer_index}"
            logger["suffix"] = f"WorkerManager-{self.consumer_index}"

        self.logger = get_logger(**logger)

        self.benchmark = kwargs["benchmark"]
        if self.benchmark and kwargs["benchmark_file"] is None:
            self.benchmark_logger = self.logger
        else:
            logger = kwargs["logger"].copy()
            logger["path"] = kwargs["benchmark_file"]
            logger["name"] = f"WorkerManager-{self.consumer_index}"
            self.benchmark_logger = get_logger(**logger)

    def extend_clarifications(self, **clarifications):
        clarifications["consumer_index"] = self.consumer_index
        clarifications["class_name"] = self.__class__.__name__
        return clarifications

    def print_log(self, message, **clarifications):
        clarifications = [f"{key}={value}" for key, value in clarifications.items()]
        clarifications = ", ".join(clarifications)
        print(f"{message} ({clarifications})")

    def debug(self, message, **clarifications):
        clarifications = self.extend_clarifications(**clarifications)
        if self.logger is None:
            self.print_log(message, **clarifications)
            return

        self.logger.debug(
            message,
            **clarifications,
        )

    def info(self, message, **clarifications):
        clarifications = self.extend_clarifications(**clarifications)
        if self.logger is None:
            self.print_log(message, **clarifications)
            return

        self.logger.info(
            message,
            **clarifications,
        )

    def warning(self, message, **clarifications):
        clarifications = self.extend_clarifications(**clarifications)
        if self.logger is None:
            self.print_log(message, **clarifications)
            return

        self.logger.warning(
            message,
            **clarifications,
        )

    def error(self, message, **clarifications):
        clarifications = self.extend_clarifications(**clarifications)
        if self.logger is None:
            self.print_log(message, **clarifications)
            return

        self.logger.warning(
            message,
            **clarifications,
        )
