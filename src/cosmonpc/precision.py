"""Explicit storage and FFT precision selection."""

import numpy as np


def complex_dtype(precision="double"):
    if precision not in ("single", "double"):
        raise ValueError("precision must be 'single' or 'double'")
    return np.dtype("complex64" if precision == "single" else "complex128")
