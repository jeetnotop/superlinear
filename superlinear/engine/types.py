"""Engine request and response types.

These types are used internally by the engine and adapters.
They are independent of any HTTP/API layer.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerateRequest:
    """Request for text generation."""

    prompt: str
    max_new_tokens: int = 4096
    temperature: float = 0.0
    stop_sequences: list[str] = field(default_factory=list)
    stream: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerateResponse:
    """Response from text generation."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"  # "stop", "length", "repetition", "error"


@dataclass
class ModelInfo:
    """Information about a loaded model."""

    model_path: str
    model_family: str
    dtype: str
    device: str
    max_context_length: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)
