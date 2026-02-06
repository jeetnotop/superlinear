# Model-family adapters
#
# Each adapter implements a common interface for:
#   - Loading model + tokenizer
#   - Running generation (streaming and non-streaming)
#   - Reporting capabilities/metadata
#
# The engine uses adapters to stay model-agnostic.

from .base import BaseAdapter

__all__ = ["BaseAdapter"]
