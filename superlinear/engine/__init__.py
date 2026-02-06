# Model-agnostic inference engine
#
# This package provides a unified interface for running inference
# across all supported model families via adapters.
#
# Key components:
#   - adapters/     Model-family specific adapters
#   - registry.py   Maps model identifiers to adapters
#   - types.py      Engine request/response types
#   - generate.py   Shared generation / streaming logic
