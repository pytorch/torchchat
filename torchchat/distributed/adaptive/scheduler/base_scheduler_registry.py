from enum import Enum
from typing import Type, Union

from chunking_scheduler import SimpleChunkingScheduler
from sarathi_scheduler import SarathiScheduler
from torchchat.distributed.adaptive.base_registry import BaseRegistry, RegistryError

from torchchat.distributed.adaptive.config.config import SchedulerType
from torchchat.distributed.logging_utils import SingletonLogger

logger = SingletonLogger.get_logger()


# scheduler1 = SchedulerRegistry.create_scheduler(SchedulerType.VLLM, **config)
# Define a common base type for all schedulers
class BaseScheduler:
    """Base class for all scheduler implementations."""

    pass


# Ensure all scheduler classes inherit from BaseScheduler
# VLLMScheduler.__bases__ = (BaseScheduler,)
# OrcaScheduler.__bases__ = (BaseScheduler,)
# FasterTransformerScheduler.__bases__ = (BaseScheduler,)
SarathiScheduler.__bases__ = (BaseScheduler,)
SimpleChunkingScheduler.__bases__ = (BaseScheduler,)


class SchedulerRegistry(BaseRegistry[SchedulerType, BaseScheduler]):
    """
    Registry for different scheduler implementations.

    This registry maintains a mapping between SchedulerType enum values
    and their corresponding scheduler implementation classes.
    """

    key_class = SchedulerType

    @classmethod
    def get_key_from_str(cls, key_str: str) -> SchedulerType:
        """
        Convert a string to its corresponding SchedulerType.

        Args:
            key_str: String representation of the scheduler type

        Returns:
            SchedulerType corresponding to the input string

        Raises:
            ValueError: If the string doesn't match any valid scheduler type
        """
        try:
            return SchedulerType.from_str(key_str)
        except ValueError as e:
            raise ValueError(f"Invalid scheduler type: {key_str}") from e

    @classmethod
    def initialize_registry(cls) -> None:
        """
        Initialize the registry with all available scheduler implementations.

        This method registers all known scheduler types with their corresponding
        implementation classes. It uses override=True to ensure idempotency.

        Raises:
            RegistryError: If there's an issue registering any scheduler
        """
        registrations = {
            # SchedulerType.VLLM: VLLMScheduler,
            # SchedulerType.ORCA: OrcaScheduler,
            # SchedulerType.FASTER_TRANSFORMER: FasterTransformerScheduler,
            # SchedulerType.SARATHI: SarathiScheduler,
            SchedulerType.SIMPLE_CHUNKING: SimpleChunkingScheduler,
        }

        for scheduler_type, scheduler_class in registrations.items():
            try:
                cls.register(scheduler_type, scheduler_class, override=True)
            except RegistryError as e:
                raise RegistryError(
                    f"Failed to register scheduler {scheduler_type}: {str(e)}"
                ) from e

    @classmethod
    def create_scheduler(
        cls, scheduler_type: Union[str, SchedulerType], *args, **kwargs
    ) -> BaseScheduler:
        """
        Create a scheduler instance based on the specified type.

        Args:
            scheduler_type: Either a SchedulerType enum value or its string representation
            *args: Positional arguments to pass to the scheduler constructor
            **kwargs: Keyword arguments to pass to the scheduler constructor

        Returns:
            An instance of the requested scheduler

        Raises:
            ValueError: If the scheduler type is invalid
            RegistryError: If the scheduler type is not registered
        """
        if isinstance(scheduler_type, str):
            return cls.get_from_str(scheduler_type, *args, **kwargs)
        elif isinstance(scheduler_type, SchedulerType):
            return cls.get(scheduler_type, *args, **kwargs)
        else:
            raise ValueError(
                f"scheduler_type must be either str or SchedulerType, got {type(scheduler_type)}"
            )

    @classmethod
    def get_available_schedulers(cls) -> set[str]:
        """
        Get a set of all available scheduler types as strings.

        Returns:
            A set of strings representing all registered scheduler types
        """
        return {key.value for key in cls.get_all_keys()}


# Initialize the registry
SchedulerRegistry.initialize_registry()
