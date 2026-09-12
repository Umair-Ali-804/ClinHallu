"""Project-root and path helpers that do not depend on the current directory."""

from __future__ import annotations

import os
from pathlib import Path


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the directory containing ``pyproject.toml``.

    ``CLINHALLU_ROOT`` can be set explicitly in unusual notebook layouts.
    """

    explicit = os.environ.get("CLINHALLU_ROOT")
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not (root / "pyproject.toml").is_file():
            raise FileNotFoundError(f"CLINHALLU_ROOT has no pyproject.toml: {root}")
        return root

    origin = Path(start or Path.cwd()).expanduser().resolve()
    if origin.is_file():
        origin = origin.parent
    for candidate in (origin, *origin.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate

    package_path = Path(__file__).resolve()
    for candidate in package_path.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate the ClinHallu project root. Run from the project "
        "directory or set CLINHALLU_ROOT."
    )


def absolute_path(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()
