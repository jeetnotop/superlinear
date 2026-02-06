"""Runtime environment checks and feature flags for superlinear."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass


@functools.lru_cache(maxsize=1)
def is_triton_available() -> bool:
    """Check if Triton is available for kernel compilation."""
    try:
        import triton
        import triton.language as tl
        return True
    except ImportError:
        return False


@functools.lru_cache(maxsize=1)
def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


@functools.lru_cache(maxsize=16)
def get_cuda_capability(device_index: int = 0) -> tuple[int, int] | None:
    """Get CUDA compute capability for a device."""
    if not is_cuda_available() or device_index >= torch.cuda.device_count():
        return None
    props = torch.cuda.get_device_properties(device_index)
    return (props.major, props.minor)


@functools.lru_cache(maxsize=16)
def get_shared_memory_limit(device_index: int = 0) -> int:
    """
    Query the available shared memory limit for a given GPU.
    
    Returns conservative 48KB fallback if CUDA is not available.
    """
    if not is_cuda_available() or device_index >= torch.cuda.device_count():
        return 48 * 1024  # Conservative 48KB fallback

    props = torch.cuda.get_device_properties(device_index)
    per_block = getattr(props, "shared_memory_per_block_optin", props.shared_memory_per_block)
    per_sm = props.shared_memory_per_multiprocessor
    return min(per_block, per_sm)


def check_triton_required() -> None:
    """Raise ImportError if Triton is not available."""
    if not is_triton_available():
        raise ImportError(
            "superlinear requires Triton for GPU kernel compilation. "
            "Install it with: pip install triton>=2.1"
        )


def check_cuda_required() -> None:
    """Raise RuntimeError if CUDA is not available."""
    if not is_cuda_available():
        raise RuntimeError(
            "superlinear requires CUDA for GPU operations. "
            "Ensure you have a CUDA-capable GPU and PyTorch with CUDA support."
        )
