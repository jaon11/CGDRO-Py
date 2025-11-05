"""CGDRO public API."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cgdro")
except PackageNotFoundError:
    __version__ = "0.0.dev0"

from .Classification import Classification
from . import Regression, data, geometry


__all__ = ["Classification", "Regression", "data", "__version__"]