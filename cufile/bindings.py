import ctypes

libcufile = ctypes.CDLL("libcufile.so")

class CUfileError(ctypes.Structure):
    _fields_ = [("err", ctypes.c_int), ("cu_err", ctypes.c_int)]

CUfileHandle_t = ctypes.c_void_p

class DescrUnion(ctypes.Union):
    _fields_ = [("fd", ctypes.c_int), ("handle", ctypes.c_void_p)]

class CUfileDescr(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("handle", DescrUnion), ("fs_ops", ctypes.c_void_p)]

libcufile.cuFileDriverOpen.restype             = CUfileError
libcufile.cuFileDriverClose.restype            = CUfileError
libcufile.cuFileHandleRegister.restype         = CUfileError
libcufile.cuFileBufRegister.restype            = CUfileError
libcufile.cuFileBufDeregister.restype          = CUfileError
libcufile.cuFileRead.restype                   = ctypes.c_size_t
libcufile.cuFileWrite.restype                  = ctypes.c_size_t
libcufile.cuFileHandleRegister.argtypes        = [ctypes.POINTER(CUfileHandle_t), ctypes.POINTER(CUfileDescr)]
libcufile.cuFileHandleDeregister.argtypes      = [CUfileHandle_t]
libcufile.cuFileBufRegister.argtypes           = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
libcufile.cuFileBufDeregister.argtypes         = [ctypes.c_void_p]
libcufile.cuFileRead.argtypes                  = [CUfileHandle_t, ctypes.c_void_p, ctypes.c_size_t,
                                                 ctypes.c_longlong, ctypes.c_longlong]
libcufile.cuFileWrite.argtypes                 = [CUfileHandle_t, ctypes.c_void_p, ctypes.c_size_t,
                                                 ctypes.c_longlong, ctypes.c_longlong]

# convenience
def _ck(status: CUfileError, name: str):
    if status.err != 0:
        raise RuntimeError(f"{name} failed (cuFile err={status.err}, cuda_err={status.cu_err})")

def cuFileDriverOpen() -> None:
    _ck(libcufile.cuFileDriverOpen(), "cuFileDriverOpen")

def cuFileDriverClose() -> None:
    _ck(libcufile.cuFileDriverClose(), "cuFileDriverClose")

def cuFileHandleRegister(fd: int) -> CUfileHandle_t:
    descr = CUfileDescr(type=1, handle=DescrUnion(fd=fd))
    handle = CUfileHandle_t()
    _ck(libcufile.cuFileHandleRegister(handle, descr), "cuFileHandleRegister")
    return handle

def cuFileHandleDeregister(handle: CUfileHandle_t) -> None:
    libcufile.cuFileHandleDeregister(handle), "cuFileHandleDeregister"

def cuFileBufRegister(buf: ctypes.c_void_p, size: int, flags: int) -> None:
    _ck(libcufile.cuFileBufRegister(buf, size, flags), "cuFileBufRegister")

def cuFileBufDeregister(buf: ctypes.c_void_p) -> None:
    _ck(libcufile.cuFileBufDeregister(buf), "cuFileBufDeregister")

def cuFileRead(handle: CUfileHandle_t, buf: ctypes.c_void_p, size: int, file_offset: int, dev_offset: int) -> int:
    return libcufile.cuFileRead(handle, buf, size, file_offset, dev_offset)

def cuFileWrite(handle: CUfileHandle_t, buf: ctypes.c_void_p, size: int, file_offset: int, dev_offset: int) -> int:
    return libcufile.cuFileWrite(handle, buf, size, file_offset, dev_offset)
