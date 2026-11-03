"""Repository path helpers.

The Python component lives in ``components/mp2-py``.  Config paths are written
relative to the repository root so that the same config works from the
component directory, the repository root, or a Nix shell.
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root(start: Path | str | None = None) -> Path:
    """Return the repository root.

    Resolution order:
    1. ``CSC869_ROOT`` environment variable, when set.
    2. Walk upwards from this file looking for ``flake.nix`` or ``data/adult``.
    3. Walk upwards from *start* (or the current directory).
    4. Fall back to the current working directory.
    """

    env = os.environ.get("CSC869_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    candidates = [Path(__file__).resolve(), Path(start).resolve() if start else Path.cwd().resolve()]
    for candidate in candidates:
        for parent in [candidate, *candidate.parents]:
            if (parent / "flake.nix").exists() or (parent / "data" / "adult").exists():
                return parent

    return Path.cwd().resolve()


def resolve(path: str | os.PathLike[str], root: Path | None = None) -> Path:
    """Resolve *path* against the repository root unless it is absolute."""

    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (root or repo_root()) / p


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
