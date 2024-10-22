 from dataclasses import dataclass, field
from threading import Lock
from typing import Optional
from contextlib import contextmanager

@dataclass
class Counter:
    """A thread-safe counter class that generates sequential numbers.
    
    This class provides a thread-safe counter implementation with support for
    custom start values, step sizes, and optional maximum values. All operations
    are atomic and safe for concurrent access.
    
    Attributes:
        start: Initial value for the counter (default: 0)
        step: Increment size for each count (default: 1)
        max_value: Optional maximum value (exclusive) for the counter
        
    Example:
        >>> counter = Counter(start=1, step=2)
        >>> next(counter)  # Thread safe
        1
        >>> counter.value  # Thread safe
        3
        >>> with counter.bulk_operations() as c:  # Atomic block
        ...     current = c.value
        ...     c.set(current + 10)
    """
    
    start: int = 0
    step: int = 1
    max_value: Optional[int] = None
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    
    def __post_init__(self) -> None:
        """Initialize counter value and validate parameters."""
        self._validate_parameters()
        self._counter = self.start
    
    def _validate_parameters(self) -> None:
        """Validate counter parameters."""
        if not isinstance(self.start, int):
            raise TypeError(f"Start value must be an integer, got {type(self.start)}")
        
        if not isinstance(self.step, int):
            raise TypeError(f"Step value must be an integer, got {type(self.step)}")
        
        if self.step == 0:
            raise ValueError("Step value cannot be zero")
            
        if self.max_value is not None:
            if not isinstance(self.max_value, int):
                raise TypeError(
                    f"Max value must be an integer, got {type(self.max_value)}"
                )
            if self.max_value <= self.start and self.step > 0:
                raise ValueError(
                    f"Max value ({self.max_value}) must be greater than "
                    f"start value ({self.start}) when step is positive"
                )
            if self.max_value >= self.start and self.step < 0:
                raise ValueError(
                    f"Max value ({self.max_value}) must be less than "
                    f"start value ({self.start}) when step is negative"
                )
    
    @contextmanager
    def bulk_operations(self) -> 'Counter':
        """Context manager for performing multiple operations atomically.
        
        This method provides a way to perform multiple operations on the counter
        while holding the lock, ensuring that no other threads can modify the
        counter during the operation.
        
        Example:
            >>> with counter.bulk_operations() as c:
            ...     if c.value < 100:
            ...         c.set(c.value + 10)
        
        Yields:
            Counter: The counter instance with exclusive access.
        """
        with self._lock:
            yield self
    
    def __next__(self) -> int:
        """Get the next value in the sequence thread-safely.
        
        Returns:
            The current counter value before incrementing.
            
        Raises:
            StopIteration: If max_value is set and would be exceeded.
        """
        with self._lock:
            current = self._counter
            next_value = current + self.step
            
            if self.max_value is not None:
                if (self.step > 0 and next_value >= self.max_value) or \
                   (self.step < 0 and next_value <= self.max_value):
                    raise StopIteration
            
            self._counter = next_value
            return current
    
    def __iter__(self) -> 'Counter':
        """Make the counter iterable.
        
        Returns:
            Self as iterator.
        """
        return self
    
    @property
    def value(self) -> int:
        """Get the current counter value thread-safely without incrementing.
        
        Returns:
            The current counter value.
        """
        with self._lock:
            return self._counter
    
    def reset(self) -> None:
        """Reset the counter to its initial value thread-safely."""
        with self._lock:
            self._counter = self.start
    
    def set(self, value: int) -> None:
        """Set the counter to a specific value thread-safely.
        
        Args:
            value: New value for the counter.
            
        Raises:
            TypeError: If value is not an integer.
            ValueError: If value exceeds max_value.
        """
        if not isinstance(value, int):
            raise TypeError(f"Value must be an integer, got {type(value)}")
        
        with self._lock:
            if self.max_value is not None:
                if (self.step > 0 and value >= self.max_value) or \
                   (self.step < 0 and value <= self.max_value):
                    raise ValueError(
                        f"Value {value} would exceed max_value {self.max_value}"
                    )
            
            self._counter = value
    
    def peek(self, n: int = 1) -> int:
        """Preview a future counter value thread-safely without modifying the counter.
        
        Args:
            n: Number of steps to look ahead (default: 1).
            
        Returns:
            The value that would be returned after n calls to next().
            
        Raises:
            ValueError: If the peeked value would exceed max_value.
        """
        with self._lock:
            future_value = self._counter + (self.step * n)
            
            if self.max_value is not None:
                if (self.step > 0 and future_value >= self.max_value) or \
                   (self.step < 0 and future_value <= self.max_value):
                    raise ValueError(
                        f"Peeked value {future_value} would exceed max_value {self.max_value}"
                    )
            
            return future_value

    def compare_and_set(self, expected: int, new_value: int) -> bool:
        """Atomically update the counter if it equals the expected value.
        
        This method provides a compare-and-swap operation that's useful for
        implementing wait-free algorithms.
        
        Args:
            expected: Value that the counter must currently equal
            new_value: Value to set if the comparison succeeds
            
        Returns:
            bool: True if the update was successful, False otherwise.
            
        Raises:
            TypeError: If either value is not an integer.
            ValueError: If new_value would exceed max_value.
        """
        if not isinstance(expected, int) or not isinstance(new_value, int):
            raise TypeError("Both expected and new values must be integers")
            
        with self._lock:
            if self._counter != expected:
                return False
                
            if self.max_value is not None:
                if (self.step > 0 and new_value >= self.max_value) or \
                   (self.step < 0 and new_value <= self.max_value):
                    raise ValueError(
                        f"New value {new_value} would exceed max_value {self.max_value}"
                    )
            
            self._counter = new_value
            return True
