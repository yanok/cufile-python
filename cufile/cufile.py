"""
Main module for CUDA file operations.
"""

import os
import ctypes
from .bindings import *

def _singleton(cls):
    _instances = {}
    def wrapper(*args, **kwargs):
        if cls not in _instances:
            _instances[cls] = cls(*args, **kwargs)
        return _instances[cls]
    return wrapper

@_singleton
class _CuFileDriver:
    def __init__(self):
        cuFileDriverOpen()
    
    def __del__(self):
        cuFileDriverClose()
    
class CuFile:
    """
    Main class for CUDA file operations.
    """
    
    def __init__(self, path: str, mode: str = "r"):
        """
        Initialize the CuFile instance.
        """
        self._driver = _CuFileDriver()
        self._path = path
        self._mode = mode
        if mode == "r":
            self._os_mode = os.O_RDONLY | os.O_DIRECT
        elif mode == "w":
            self._os_mode = os.O_CREAT | os.O_WRONLY | os.O_TRUNC | os.O_DIRECT
        else:
            assert mode == "rw"
            self._os_mode = os.O_RDWR | os.O_DIRECT

    def __enter__(self):
        """
        Context manager entry.
        """
        self._handle = os.open(self._path, self._os_mode)
        self._cu_file_handle = cuFileHandleRegister(self._handle)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit.
        """
        cuFileHandleDeregister(self._cu_file_handle)
        os.close(self._handle)
    
    def read(self, dest: ctypes.c_void_p, size: int, file_offset: int = 0, dev_offset: int = 0):
        """
        Read from the file.
        """
        return cuFileRead(self._cu_file_handle, dest, size, file_offset, dev_offset)
    
    def write(self, src: ctypes.c_void_p, size: int, file_offset: int = 0, dev_offset: int = 0):
        """
        Write to the file.
        """
        return cuFileWrite(self._cu_file_handle, src, size, file_offset, dev_offset)