"""Base adapter interface for model families."""

from abc import ABC, abstractmethod
from typing import Any, Iterator


class BaseAdapter(ABC):
    """
    Abstract base class for model-family adapters.
    
    Each supported model family implements this interface so the engine
    can run inference without knowing model-specific details.
    """

    @abstractmethod
    def load(self, model_path: str, **kwargs) -> None:
        """
        Load model and tokenizer from the given path or HF repo.
        
        Args:
            model_path: Local path or HF Hub model identifier.
            **kwargs: Model-specific loading options (dtype, device, etc.).
        """
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        stop_sequences: list[str] | None = None,
        **kwargs,
    ) -> str:
        """
        Generate a completion for the given prompt.
        
        Args:
            prompt: Input text.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            stop_sequences: Sequences that stop generation.
            **kwargs: Additional generation parameters.
        
        Returns:
            Generated text (excluding the prompt).
        """
        pass

    @abstractmethod
    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        stop_sequences: list[str] | None = None,
        **kwargs,
    ) -> Iterator[str]:
        """
        Stream tokens for the given prompt.
        
        Args:
            prompt: Input text.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            stop_sequences: Sequences that stop generation.
            **kwargs: Additional generation parameters.
        
        Yields:
            Generated tokens one at a time.
        """
        pass

    @property
    @abstractmethod
    def model_info(self) -> dict[str, Any]:
        """
        Return metadata about the loaded model.
        
        Returns:
            Dict with keys like 'model_path', 'dtype', 'max_context_length', etc.
        """
        pass

    def unload(self) -> None:
        """
        Unload the model and free resources.
        
        Default implementation does nothing; override if cleanup is needed.
        """
        pass
