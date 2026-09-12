"""Common baseline specification."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    module: str
    arguments: tuple[str, ...]
