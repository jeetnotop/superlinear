import os
import sys


# Ensure the repository root is on sys.path so tests can import local entrypoints
# like apps.server.app without requiring an editable install.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
