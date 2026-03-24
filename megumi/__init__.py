"""megumi: a feature selection toolkit."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("megumi")
except PackageNotFoundError:
    __version__ = "unknown"
