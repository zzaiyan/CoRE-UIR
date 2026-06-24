"""Helpers for running BasicSR entry scripts directly from the repo tree.

When a script is launched as ``python basicsr/train.py``, Python only adds the
``basicsr`` directory to ``sys.path``. This helper prepends the repository root
so absolute imports like ``import basicsr`` and ``import prior`` work
without hard-coded machine-specific paths.
"""

from pathlib import Path
import sys


def setup_project_root(current_file):
    """Prepend the repository root inferred from ``current_file`` to ``sys.path``."""
    project_root = Path(current_file).resolve().parents[1]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root
