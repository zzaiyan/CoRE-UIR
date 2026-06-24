"""Helpers for running prior scripts directly from the repo tree."""

from pathlib import Path
import sys


def setup_project_root(current_file):
    """Prepend the repository root inferred from ``current_file`` to ``sys.path``."""
    project_root = Path(current_file).resolve().parents[1]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root
