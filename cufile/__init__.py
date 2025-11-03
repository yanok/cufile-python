"""
cufile-python - A basic Python wrapper for the NVidia cuFile API
"""

__version__ = "0.1.0"

from .cufile import CuFile, CuFileDriver

__all__ = ["CuFile", "CuFileDriver"]
