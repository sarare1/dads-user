"""Cooperative cancellation for training jobs.

Training runs in a worker thread, and a Python thread can't be killed from outside. The API sets
this flag ("Stop training"); the training loops call `check()` every batch and unwind by raising
`TrainingCancelled`, which the job handler turns into a clean "cancelled" status (nothing saved).
"""
import threading


class TrainingCancelled(Exception):
    """Raised inside a training loop when the user pressed Stop."""


_event = threading.Event()


def reset() -> None:
    _event.clear()


def request() -> None:
    _event.set()


def requested() -> bool:
    return _event.is_set()


def check() -> None:
    if _event.is_set():
        raise TrainingCancelled("Training was stopped by the user.")
