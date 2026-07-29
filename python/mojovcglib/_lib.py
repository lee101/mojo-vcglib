"""ctypes loader for the compiled Mojo kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "vcglib.mojo")
LIB = os.path.join(ROOT, "dist", "libmojo-vcglib.so")

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mvc_face_validity_f64": ([I, I, I, I, I, F], I),
    "mvc_edge_incidence_stats_i64": ([I, I, I], None),
    "mvc_laplacian_f64": ([I] * 9, None),
    "mvc_taubin_f64": ([I] * 9 + [F, F], None),
    "mvc_hc_f64": ([I] * 10, None),
    "mvc_scale_dependent_f64": ([I] * 9 + [F], None),
    "mvc_loop_vertices_f64": ([I] * 9, None),
    "mvc_mass_properties_f64": ([I, I, I, I], None),
    "mvc_montecarlo_f64": ([I, I, I, I, I, I, I], I),
    "mvc_stratified_f64": ([I, I, I, I, I, I], I),
    "mvc_closest_points_f64": ([I] * 8, None),
    "mvc_ray_mesh_f64": ([I] * 9, None),
    "mvc_geodesic_f64": ([I] * 12 + [F], I),
    "mvc_qem_init_f64": ([I, I, I, I, I, I, F, I], None),
    "mvc_qem_evaluate_f64": ([I, I, I, I, I, I, F], None),
    "mvc_qem_decimate_unconstrained_f64": ([I] * 9 + [F], None),
    "mvc_version": ([], I),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if (
        not force
        and os.path.exists(LIB)
        and os.path.getmtime(LIB) >= os.path.getmtime(SRC)
    ):
        return LIB
    mojo = shutil.which("mojo")
    if not mojo:
        raise BuildError("mojo not found; run through `pixi run`")
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    proc = subprocess.run(
        [mojo, "build", "--emit", "shared-lib", SRC, "-o", LIB],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip())
    return LIB


_library: ctypes.CDLL | None = None
_EMPTY_ADDRESS_SENTINEL = np.empty(1, dtype=np.float64)


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def f64(value, *, copy: bool = False) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        raise TypeError("coordinates must contain real numeric values")
    converted = np.asarray(array, dtype=np.float64)
    if array.dtype.kind in "iu" and array.size:
        # Float64 cannot distinguish adjacent integers outside this range.
        if np.any(array > 2**53) or np.any(array < -(2**53)):
            raise ValueError("integer coordinate cannot be represented exactly as float64")
    elif array.dtype.kind == "f" and array.dtype.itemsize > 8 and array.size:
        if not np.array_equal(array, converted.astype(array.dtype)):
            raise ValueError("coordinate cannot be represented exactly as float64")
    return np.array(converted, order="C", copy=True) if copy else np.ascontiguousarray(converted)


def i64(value, *, copy: bool = False) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "biu":
        raise TypeError("indices must contain integers")
    if array.dtype.kind == "u" and array.size:
        if np.any(array > np.iinfo(np.int64).max):
            raise ValueError("index is outside the int64 range")
    converted = np.asarray(array, dtype=np.int64)
    return np.array(converted, order="C", copy=True) if copy else np.ascontiguousarray(converted)


def addr(array: np.ndarray) -> int:
    if not array.flags.c_contiguous:
        raise ValueError("FFI buffers must be C-contiguous")
    address = int(array.ctypes.data)
    # Mojo UnsafePointer is non-nullable even when a zero-length buffer will
    # never be dereferenced.
    return address or int(_EMPTY_ADDRESS_SENTINEL.ctypes.data)
