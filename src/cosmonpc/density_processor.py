"""Load scalar meshes without particle painting or Poisson corrections.

HDF5 reads are restricted to each rank's mesh domain. Tile corners and cell
sizes specify placement; filenames and tile enumeration order are irrelevant.
"""

import os
import numpy as np
import h5py
from mpi4py import MPI
from pmesh import ParticleMesh
from .precision import complex_dtype


def _descriptor(source):
    if isinstance(source, np.ndarray):
        return {"array": source, "kind": "delta"}
    if isinstance(source, (str, os.PathLike)):
        return {"path": source, "kind": "delta"}
    if not isinstance(source, dict):
        raise TypeError("A field must be an ndarray, HDF5 path, or field descriptor")
    return source


def read_local_field(source, nmesh, boxsize, origin, slices, dtype):
    """Read one rectangular domain, validating complete, nonoverlapping coverage.

    Returns raw values and the descriptor. Normalization is performed globally
    by get_density_meshes, never independently for individual tiles.
    """
    spec = _descriptor(source)
    if sum(key in spec for key in ("array", "path", "tiles")) != 1:
        raise ValueError("Specify exactly one of array, path, or tiles")
    kind = spec.get("kind", "density" if "tiles" in spec else "delta")
    if kind not in ("delta", "density"):
        raise ValueError("Field kind must be 'delta' or 'density'")
    nmesh = np.asarray(nmesh, dtype=int)
    boxsize, origin = np.asarray(boxsize, float), np.asarray(origin, float)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("field_origin must contain three finite coordinates")
    start = np.array([s.start for s in slices])
    stop = np.array([s.stop for s in slices])
    shape = tuple(stop - start)
    out = np.empty(shape, dtype=dtype)
    if "array" in spec:
        data = spec["array"]
        if not isinstance(data, np.ndarray) or data.shape != tuple(nmesh):
            raise ValueError("In-memory fields must be global arrays with shape nmesh on every rank")
        if data.dtype.kind not in "fiu":
            raise TypeError("Density values must be real numeric scalars")
        out[...] = data[slices]
    else:
        paths = spec["tiles"] if "tiles" in spec else [spec["path"]]
        if isinstance(paths, (str, os.PathLike)) or not paths:
            raise ValueError("tiles must be a nonempty list of HDF5 paths")
        covered = np.zeros(shape, dtype=bool)
        cellsize = boxsize / nmesh
        times = []
        for path in paths:
            with h5py.File(path, "r") as hf:
                def meta(key):
                    return hf.attrs["/info/scalars/" + key]

                counts_raw = np.array([meta(f"N{i}") for i in range(3)])
                counts = counts_raw.astype(int)
                lengths = np.array([meta(f"L{i}") for i in range(3)], float)
                corner = np.array([meta(f"corner{i}") for i in range(3)], float)
                if np.any(counts <= 0) or np.any(counts != counts_raw):
                    raise ValueError(f"Invalid field dimensions in {path}")
                if not np.allclose(lengths / counts, cellsize, rtol=1e-7, atol=1e-10):
                    raise ValueError(f"Field cell size does not match the configured mesh: {path}")
                offset = (corner - origin) / cellsize
                if not np.all(np.isfinite(offset)) or not np.allclose(offset, np.rint(offset), rtol=0, atol=1e-5):
                    raise ValueError(f"Field corner is not aligned with the configured mesh: {path}")
                offset = np.rint(offset).astype(int)
                if np.any(offset < 0) or np.any(offset + counts > nmesh):
                    raise ValueError(f"Field extends outside the configured box: {path}")
                dataset = hf[spec.get("dataset", "/scalars/field")]
                if dataset.shape != tuple(counts) or int(meta("rank")) not in (0, 1):
                    raise ValueError(f"Expected a 3D scalar Field in {path}")
                if dataset.dtype.kind not in "fiu":
                    raise TypeError(f"Expected real numeric density values in {path}")
                time_key = "/info/scalars/time"
                if time_key in hf.attrs:
                    times.append(float(hf.attrs[time_key]))
                lo, hi = np.maximum(start, offset), np.minimum(stop, offset + counts)
                if np.any(lo >= hi):
                    continue
                target = tuple(slice(int(a), int(b)) for a, b in zip(lo - start, hi - start))
                source_slice = tuple(slice(int(a), int(b)) for a, b in zip(lo - offset, hi - offset))
                if np.any(covered[target]):
                    raise ValueError("Overlapping density tiles")
                out[target] = dataset[source_slice]
                covered[target] = True
        if times and not np.allclose(times, times[0], rtol=1e-10, atol=0):
            raise ValueError("Density tiles have inconsistent cosmic times")
        if not np.all(covered):
            raise ValueError("Density fields do not completely cover the configured mesh")
    if not np.all(np.isfinite(out)):
        raise ValueError("Density field contains nonfinite values")
    if kind == "density" and np.any(out < 0):
        raise ValueError("Density values must be nonnegative; use kind='delta' for overdensity")
    return out, {**spec, "kind": kind}


def get_density_meshes(fields, correlation_mode, nmesh, boxsize, comm, statistic,
                       tracer_type="ab", precision="double", field_origin=(0., 0., 0.),
                       **kwargs):
    """Adapt continuous fields to the existing box estimators' mesh convention."""
    pm = ParticleMesh(BoxSize=boxsize, Nmesh=nmesh, dtype=complex_dtype(precision), comm=comm)
    labels = ["a"] + (["b"] if correlation_mode == "cross" else [])
    if statistic == "bk_sugi" and tracer_type == "abc":
        labels.append("c")
    meshes, attrs = {}, {"input_type": "density", "field_metadata": {}}
    cells = int(np.prod(nmesh))
    volume = float(np.prod(boxsize))
    for label in labels:
        mesh = pm.create(type="real")
        error, values, spec = None, None, None
        try:
            values, spec = read_local_field(fields[f"data_{label}"], nmesh, boxsize,
                                           field_origin, mesh.slices, mesh.value.real.dtype)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        errors = comm.allgather(error)
        if any(errors):
            raise ValueError("Cannot load density field: " + next(e for e in errors if e))
        mean = comm.allreduce(float(np.sum(values, dtype=np.float64)), op=MPI.SUM) / cells
        if spec["kind"] == "density":
            reference = float(spec.get("mean_density", mean))
            if not np.isfinite(reference) or reference <= 0:
                raise ValueError("mean_density must be finite and positive")
            values /= reference
            values -= 1
        # Always remove the DC component. For pk this is also explicitly cleared
        # in Fourier space, but bk must see a zero-mean third leg as well.
        delta_mean = comm.allreduce(float(np.sum(values, dtype=np.float64)), op=MPI.SUM) / cells
        values -= delta_mean
        # Keep delta itself to preserve small-amplitude fields in single
        # precision (adding and later subtracting one would lose those modes).
        mesh[...] = values
        meshes[label] = mesh
        # Virtual unit cell weights preserve the existing estimator normalization.
        attrs[f"NZ_{label}"] = cells / volume
        attrs[f"N_gal_{label}"] = cells
        attrs["field_metadata"][label] = {
            "kind": spec["kind"], "input_mean": mean,
            "mean_density": reference if spec["kind"] == "density" else None,
            "removed_delta_mean": delta_mean,
        }
    if statistic == "pk":
        return attrs, meshes["a"], meshes.get("b")
    return attrs, meshes["a"], meshes.get("b"), meshes.get("c")
