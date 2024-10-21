import os
import sys
import traceback
from functools import wraps
from threading import Lock
from typing import Any, Callable, TypeVar

T = TypeVar('T')

def synchronized(method: Callable[..., T]) -> Callable[..., T]:
    """
    Synchronization decorator at the instance level.
    
    This decorator ensures that the method is thread-safe by using a Lock.
    """
    lock_attr = f"_lock_{method.__name__}"

    @wraps(method)
    def synced_method(self: Any, *args: Any, **kwargs: Any) -> T:
        if not hasattr(self, lock_attr):
            setattr(self, lock_attr, Lock())
        with getattr(self, lock_attr):
            return method(self, *args, **kwargs)

    return synced_method

def exit_on_error(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to exit the program on unhandled exceptions.
    
    This decorator will print the traceback and exit the program if an unhandled exception occurs.
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        try:
            return func(*args, **kwargs)
        except Exception:
            traceback.print_exc()
            sys.exit(1)

    return wrapper
