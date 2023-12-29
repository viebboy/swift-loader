"""
signals.py: signals definition between parent and child process
---------------------------------------------------------------


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

from enum import Enum


class PARENT_MESSAGE(Enum):
    """
    List of messages sent from parent
    """

    ACK = 1
    TERMINATE = 2

    def all():
        variables = []
        class_ = globals()[__class__.__name__]
        for x in dir(class_):
            if not x.startswith("__") and not callable(getattr(class_, x)):
                variables.append(getattr(class_, x))
        return variables

    def decode(value):
        class_ = globals()[__class__.__name__]
        all_func = getattr(class_, "all")
        for e in all_func():
            if e.value == value:
                return e


class CHILD_MESSAGE(Enum):
    """
    List of messages sent from child
    """

    EPOCH_END = 1
    TERMINATE = 2
    METADATA = 3

    def all():
        variables = []
        class_ = globals()[__class__.__name__]
        for x in dir(class_):
            if not x.startswith("__") and not callable(getattr(class_, x)):
                variables.append(getattr(class_, x))
        return variables

    def decode(value):
        class_ = globals()[__class__.__name__]
        all_func = getattr(class_, "all")
        for e in all_func():
            if e.value == value:
                return e
