"""Signal definitions for communication between parent and child processes.

This module defines message types used for inter-process communication between
the parent process (main process) and child processes (worker processes).

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

from enum import Enum


class PARENT_MESSAGE(Enum):
    """Messages sent from parent process to child processes.

    Attributes:
        ACK: Acknowledgment message.
        TERMINATE: Termination signal to stop the child process.
    """

    ACK = 1
    TERMINATE = 2

    @staticmethod
    def all() -> list[PARENT_MESSAGE]:
        """Get all message types.

        Returns:
            List of all PARENT_MESSAGE enum values.
        """
        variables = []
        class_ = globals()[__class__.__name__]
        for x in dir(class_):
            if not x.startswith("__") and not callable(getattr(class_, x)):
                variables.append(getattr(class_, x))
        return variables

    @staticmethod
    def decode(value: int) -> PARENT_MESSAGE:
        """Decode integer value to PARENT_MESSAGE enum.

        Args:
            value: Integer value to decode.

        Returns:
            Corresponding PARENT_MESSAGE enum value.

        Raises:
            ValueError: If value does not correspond to any message type.
        """
        class_ = globals()[__class__.__name__]
        all_func = getattr(class_, "all")
        for e in all_func():
            if e.value == value:
                return e
        raise ValueError(f"Invalid PARENT_MESSAGE value: {value}")


class CHILD_MESSAGE(Enum):
    """Messages sent from child processes to parent process.

    Attributes:
        EPOCH_END: Signal indicating end of an epoch.
        TERMINATE: Termination signal indicating child process is stopping.
        METADATA: Metadata message containing dataset information.
    """

    EPOCH_END = 1
    TERMINATE = 2
    METADATA = 3

    @staticmethod
    def all() -> list[CHILD_MESSAGE]:
        """Get all message types.

        Returns:
            List of all CHILD_MESSAGE enum values.
        """
        variables = []
        class_ = globals()[__class__.__name__]
        for x in dir(class_):
            if not x.startswith("__") and not callable(getattr(class_, x)):
                variables.append(getattr(class_, x))
        return variables

    @staticmethod
    def decode(value: int) -> CHILD_MESSAGE:
        """Decode integer value to CHILD_MESSAGE enum.

        Args:
            value: Integer value to decode.

        Returns:
            Corresponding CHILD_MESSAGE enum value.

        Raises:
            ValueError: If value does not correspond to any message type.
        """
        class_ = globals()[__class__.__name__]
        all_func = getattr(class_, "all")
        for e in all_func():
            if e.value == value:
                return e
        raise ValueError(f"Invalid CHILD_MESSAGE value: {value}")
