"""
Tests for the cufile module.
"""

import os
import ctypes
import pytest
import time
from cuda.bindings import driver as cuda
from cufile import CuFile

BUF_SIZE = int(os.environ.get("TEST_CUFILE_BUF_SIZE", 256)) * 1024 * 1024
WORK_DIR = os.environ.get("TEST_CUFILE_WORK_DIR", ".")
PATTERN_BYTE = int(os.environ.get("TEST_CUFILE_PATTERN_BYTE", 0xAB))
CUDA_DEVICE = int(os.environ.get("TEST_CUFILE_CUDA_DEVICE", 0))

file_path = os.path.join(WORK_DIR, "test.bin")

err, = cuda.cuInit(0)
assert err == 0, f"cuInit failed: {err}"
err, device = cuda.cuDeviceGet(CUDA_DEVICE)
assert err == 0, f"cuDeviceGet failed: {err}"
params = cuda.CUctxCreateParams()
assert params is not None
err, context = cuda.cuCtxCreate(params, 0, device)
assert err == 0, f"cuCtxCreate failed: {err}"
err, dptr_w = cuda.cuMemAlloc(BUF_SIZE)
assert err == 0, f"cuMemAlloc failed: {err}"
err, dptr_r = cuda.cuMemAlloc(BUF_SIZE)
assert err == 0, f"cuMemAlloc failed: {err}"
err, hptr = cuda.cuMemAllocHost(BUF_SIZE)
assert err == 0, f"cuMemAllocHost failed: {err}"
err, = cuda.cuMemsetD8(dptr_w, PATTERN_BYTE, BUF_SIZE)
assert err == 0, f"cuMemsetD8 failed: {err}"

def test_cufile_initialization():
    """Test that CuFile can be initialized."""
    cufile = CuFile(file_path, "w")
    assert isinstance(cufile, CuFile)

def test_cufile_context_manager():
    """Test that CuFile works as a context manager."""
    with CuFile(file_path, "w") as cufile:
        assert isinstance(cufile, CuFile)

def test_cufile_read_write_with_context_manager():
    """Test that CuFile can read and write to a file."""
    begin = time.perf_counter()
    with CuFile(file_path, "w") as cufile:
        begin_write = time.perf_counter()
        ret = cufile.write(ctypes.c_void_p(int(dptr_w)), BUF_SIZE)
        write_time = time.perf_counter() - begin_write
        assert ret == BUF_SIZE
    dt = time.perf_counter() - begin
    print(f"WRITE (w/o open/register) {ret/1024/1024:.2f}MB in {write_time*1e3:.2f}ms ({ret/write_time/1024/1024/1024:.2f}GB/s)")
    print(f"FULL WRITE {ret/1024/1024:.2f}MB in {dt*1e3:.2f}ms ({ret/dt/1024/1024/1024:.2f}GB/s)")

    begin = time.perf_counter()
    with CuFile(file_path, "r") as cufile:
        begin_read = time.perf_counter()
        ret = cufile.read(ctypes.c_void_p(int(dptr_r)), BUF_SIZE)
        read_time = time.perf_counter() - begin_read
        assert ret == BUF_SIZE
    dt = time.perf_counter() - begin
    print(f"READ (w/o open/register) {ret/1024/1024:.2f}MB in {read_time*1e3:.2f}ms ({ret/read_time/1024/1024/1024:.2f}GB/s)")
    print(f"FULL READ {ret/1024/1024:.2f}MB in {dt*1e3:.2f}ms ({ret/dt/1024/1024/1024:.2f}GB/s)")

    err, = cuda.cuMemcpyDtoH(hptr, dptr_r, BUF_SIZE)
    assert err == 0, f"cuMemcpyDtoH failed: {err}"
    host_buf = (ctypes.c_ubyte * BUF_SIZE).from_address(hptr)
    for i in range(BUF_SIZE):
        assert host_buf[i] == PATTERN_BYTE

def test_cufile_read_write():
    """Test that CuFile can read and write to a file."""

    begin = time.perf_counter()
    cufile = CuFile(file_path, "w")
    cufile.open()
    begin_write = time.perf_counter()
    ret = cufile.write(ctypes.c_void_p(int(dptr_w)), BUF_SIZE)
    write_time = time.perf_counter() - begin_write
    assert ret == BUF_SIZE
    cufile.close()
    dt = time.perf_counter() - begin
    print(f"WRITE (w/o open/register) {ret/1024/1024:.2f}MB in {write_time*1e3:.2f}ms ({ret/write_time/1024/1024/1024:.2f}GB/s)")
    print(f"FULL WRITE {ret/1024/1024:.2f}MB in {dt*1e3:.2f}ms ({ret/dt/1024/1024/1024:.2f}GB/s)")

    begin = time.perf_counter()
    cufile = CuFile(file_path, "r")
    cufile.open()
    begin_read = time.perf_counter()
    ret = cufile.read(ctypes.c_void_p(int(dptr_r)), BUF_SIZE)
    read_time = time.perf_counter() - begin_read
    assert ret == BUF_SIZE
    cufile.close()
    dt = time.perf_counter() - begin
    print(f"READ (w/o open/register) {ret/1024/1024:.2f}MB in {read_time*1e3:.2f}ms ({ret/read_time/1024/1024/1024:.2f}GB/s)")
    print(f"FULL READ {ret/1024/1024:.2f}MB in {dt*1e3:.2f}ms ({ret/dt/1024/1024/1024:.2f}GB/s)")

    err, = cuda.cuMemcpyDtoH(hptr, dptr_r, BUF_SIZE)
    assert err == 0, f"cuMemcpyDtoH failed: {err}"
    host_buf = (ctypes.c_ubyte * BUF_SIZE).from_address(hptr)
    for i in range(BUF_SIZE):
        assert host_buf[i] == PATTERN_BYTE

def test_read_write_without_open():
    cufile = CuFile(file_path, "r")
    try:
        cufile.read(ctypes.c_void_p(int(dptr_r)), BUF_SIZE)
        raise AssertionError("Expected an exception but none was raised")
    except IOError as e:
        assert "File is not open." in str(e)
    finally:
        cufile.close()

    cufile = CuFile(file_path, "w")
    try:
        cufile.write(ctypes.c_void_p(int(dptr_w)), BUF_SIZE)
        raise AssertionError("Expected an exception but none was raised")
    except IOError as e:
        assert "File is not open." in str(e)
    finally:
        cufile.close()
