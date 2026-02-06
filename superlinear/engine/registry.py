"""Model adapter registry.

Maps model identifiers to their corresponding adapter classes.
"""

from typing import Type

from .adapters.base import BaseAdapter
from .adapters.superlinear import SuperlinearAdapter

# Registry mapping model family names to adapter classes
_ADAPTER_REGISTRY: dict[str, Type[BaseAdapter]] = {
    "superlinear": SuperlinearAdapter,
}


def get_adapter(model_family: str) -> BaseAdapter:
    """
    Get an adapter instance for the given model family.
    
    Args:
        model_family: Name of the model family (e.g., "superlinear").
    
    Returns:
        An adapter instance for the model family.
    
    Raises:
        ValueError: If the model family is not registered.
    """
    if model_family not in _ADAPTER_REGISTRY:
        available = ", ".join(_ADAPTER_REGISTRY.keys())
        raise ValueError(
            f"Unknown model family: {model_family!r}. Available: {available}"
        )
    return _ADAPTER_REGISTRY[model_family]()


def register_adapter(model_family: str, adapter_cls: Type[BaseAdapter]) -> None:
    """
    Register a new adapter for a model family.
    
    Args:
        model_family: Name of the model family.
        adapter_cls: Adapter class (must inherit from BaseAdapter).
    """
    _ADAPTER_REGISTRY[model_family] = adapter_cls


def list_model_families() -> list[str]:
    """Return list of registered model family names."""
    return list(_ADAPTER_REGISTRY.keys())
