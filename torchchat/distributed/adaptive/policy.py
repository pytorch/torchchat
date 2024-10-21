from abc import ABC, abstractmethod
from typing import List, Dict, Type
from sarathi.core.datatypes.sequence import Sequence

class Policy(ABC):
    @abstractmethod
    def get_priority(self, now: float, seq: Sequence) -> float:
        """
        Calculate the priority of a sequence at a given time.

        Args:
            now (float): The current timestamp.
            seq (Sequence): The sequence to prioritize.

        Returns:
            float: The priority value.
        """
        pass

    def sort_by_priority(self, now: float, seqs: List[Sequence]) -> List[Sequence]:
        """
        Sort sequences by their priority.

        Args:
            now (float): The current timestamp.
            seqs (List[Sequence]): The sequences to sort.

        Returns:
            List[Sequence]: Sorted list of sequences.
        """
        return sorted(
            seqs,
            key=lambda seq: self.get_priority(now, seq),
            reverse=True,
        )

class FCFS(Policy):
    def get_priority(self, now: float, seq: Sequence) -> float:
        """
        Calculate priority based on First-Come-First-Served principle.

        Args:
            now (float): The current timestamp.
            seq (Sequence): The sequence to prioritize.

        Returns:
            float: The priority value (waiting time).
        """
        return now - seq.arrival_time

class PolicyFactory:
    _POLICY_REGISTRY: Dict[str, Type[Policy]] = {
        "fcfs": FCFS,
    }

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy:
        """
        Get a policy instance by name.

        Args:
            policy_name (str): The name of the policy to instantiate.
            **kwargs: Additional arguments to pass to the policy constructor.

        Returns:
            Policy: An instance of the requested policy.

        Raises:
            KeyError: If the policy name is not found in the registry.
        """
        try:
            policy_class = cls._POLICY_REGISTRY[policy_name]
            return policy_class(**kwargs)
        except KeyError:
            raise ValueError(f"Unknown policy: {policy_name}")

    @classmethod
    def register_policy(cls, name: str, policy_class: Type[Policy]) -> None:
        """
        Register a new policy class.

        Args:
            name (str): The name to register the policy under.
            policy_class (Type[Policy]): The policy class to register.

        Raises:
            ValueError: If a policy with the given name is already registered.
        """
        if name in cls._POLICY_REGISTRY:
            raise ValueError(f"Policy {name} is already registered")
        cls._POLICY_REGISTRY[name] = policy_class
