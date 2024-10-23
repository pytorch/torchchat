from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Type, TypeVar, Generic, Optional

# Generic type variables
T = TypeVar('T', bound=Enum)  # For the key type
V = TypeVar('V')  # For the value type

"""
from enum import Enum

class ServiceType(Enum):
    HTTP = "http"
    GRPC = "grpc"

class ServiceRegistry(BaseRegistry[ServiceType, Any]):
    key_class = ServiceType
    
    @classmethod
    def get_key_from_str(cls, key_str: str) -> ServiceType:
        try:
            return ServiceType(key_str.lower())
        except ValueError:
            raise ValueError(f"Invalid service type: {key_str}")

# Usage
class HttpService:
    def __init__(self, host: str):
        self.host = host

ServiceRegistry.register(ServiceType.HTTP, HttpService)
service = ServiceRegistry.get(ServiceType.HTTP, host="localhost")

"""


class RegistryError(Exception):
    """Custom exception for registry-related errors."""
    pass

class BaseRegistry(ABC, Generic[T, V]):
    """
    Abstract base class for implementing a registry pattern with enum keys.
    
    This class provides a type-safe registry mechanism where implementations
    can be registered and retrieved using enum keys.
    
    Type Parameters:
        T: The enum type used for keys (must be a subclass of Enum)
        V: The type of values stored in the registry
    """
    
    key_class: Type[T]  # Type of enum to use as keys
    _registry: Dict[T, Type[V]]  # Internal registry storage
    
    def __init_subclass__(cls, **kwargs):
        """Initialize the registry storage for each subclass."""
        super().__init_subclass__(**kwargs)
        cls._registry = {}
    
    @classmethod
    def register(
        cls, 
        key: T, 
        implementation_class: Type[V],
        override: bool = False
    ) -> None:
        """
        Register an implementation class for a given key.
        
        Args:
            key: Enum key to register
            implementation_class: Class to register for the key
            override: If True, allows overriding existing registrations
        
        Raises:
            RegistryError: If key is already registered and override is False
            TypeError: If key is not an instance of key_class
        """
        if not isinstance(key, cls.key_class):
            raise TypeError(f"Key must be an instance of {cls.key_class.__name__}")
            
        if key in cls._registry and not override:
            raise RegistryError(
                f"Key {key} is already registered. Use override=True to force registration."
            )
            
        cls._registry[key] = implementation_class
    
    @classmethod
    def unregister(cls, key: T) -> None:
        """
        Remove a registration for a given key.
        
        Args:
            key: Enum key to unregister
            
        Raises:
            RegistryError: If key is not registered
            TypeError: If key is not an instance of key_class
        """
        if not isinstance(key, cls.key_class):
            raise TypeError(f"Key must be an instance of {cls.key_class.__name__}")
            
        if key not in cls._registry:
            raise RegistryError(f"Key {key} is not registered")
            
        del cls._registry[key]
    
    @classmethod
    def get(cls, key: T, *args, **kwargs) -> V:
        """
        Create an instance of the registered class for a given key.
        
        Args:
            key: Enum key to look up
            *args: Positional arguments to pass to the constructor
            **kwargs: Keyword arguments to pass to the constructor
            
        Returns:
            An instance of the registered class
            
        Raises:
            RegistryError: If key is not registered
            TypeError: If key is not an instance of key_class
        """
        if not isinstance(key, cls.key_class):
            raise TypeError(f"Key must be an instance of {cls.key_class.__name__}")
            
        if key not in cls._registry:
            raise RegistryError(f"Key {key} is not registered")
            
        return cls._registry[key](*args, **kwargs)
    
    @classmethod
    def get_class(cls, key: T) -> Type[V]:
        """
        Get the registered class for a given key without instantiating it.
        
        Args:
            key: Enum key to look up
            
        Returns:
            The registered class
            
        Raises:
            RegistryError: If key is not registered
            TypeError: If key is not an instance of key_class
        """
        if not isinstance(key, cls.key_class):
            raise TypeError(f"Key must be an instance of {cls.key_class.__name__}")
            
        if key not in cls._registry:
            raise RegistryError(f"Key {key} is not registered")
            
        return cls._registry[key]
    
    @classmethod
    @abstractmethod
    def get_key_from_str(cls, key_str: str) -> T:
        """
        Convert a string representation to the corresponding enum key.
        
        Args:
            key_str: String representation of the key
            
        Returns:
            The corresponding enum key
            
        Raises:
            ValueError: If the string cannot be converted to a valid key
        """
        pass
    
    @classmethod
    def get_from_str(cls, key_str: str, *args, **kwargs) -> V:
        """
        Create an instance using a string representation of the key.
        
        Args:
            key_str: String representation of the key
            *args: Positional arguments to pass to the constructor
            **kwargs: Keyword arguments to pass to the constructor
            
        Returns:
            An instance of the registered class
            
        Raises:
            ValueError: If the string cannot be converted to a valid key
            RegistryError: If the converted key is not registered
        """
        return cls.get(cls.get_key_from_str(key_str), *args, **kwargs)
    
    @classmethod
    def is_registered(cls, key: T) -> bool:
        """
        Check if a key is registered.
        
        Args:
            key: Enum key to check
            
        Returns:
            True if the key is registered, False otherwise
            
        Raises:
            TypeError: If key is not an instance of key_class
        """
        if not isinstance(key, cls.key_class):
            raise TypeError(f"Key must be an instance of {cls.key_class.__name__}")
            
        return key in cls._registry
    
    @classmethod
    def get_all_keys(cls) -> set[T]:
        """
        Get all registered keys.
        
        Returns:
            A set of all registered keys
        """
        return set(cls._registry.keys())
