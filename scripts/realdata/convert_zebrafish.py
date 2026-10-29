# The zebrafish trajectories of Heras et al. (2019), idtracker.ai pickles (.npy holding a dict) -> HDF5 for
# zebrafish.jl, with an allow-listed unpickler (the files only reference numpy's array reconstruction).
#
# usage: python -I convert_zebrafish.py <out_dir> <trajectories.npy>...
# out_dir/<name>.h5: positions (frames, fish, 2) in px (NaN where missing), attrs frames_per_second,
# body_length (px), and setup_points/<key> (px) if present. Name the files zebrafish_<fish>_<video>.npy.
import io
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import numpy.lib.format as fmt

ALLOW = {("_codecs", "encode"), ("numpy", "dtype"), ("numpy", "ndarray"),
         ("numpy.core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"),
         ("numpy._core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "scalar")}


class Safe(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOW:
            raise pickle.UnpicklingError(f"refused {module}.{name}")
        if module == "numpy.core.multiarray":
            module = "numpy._core.multiarray"
        return super().find_class(module, name)


def load(path):
    with open(path, "rb") as h:
        version = fmt.read_magic(h)
        (fmt.read_array_header_1_0 if version == (1, 0) else fmt.read_array_header_2_0)(h)
        d = Safe(h, encoding="latin1").load()
    return d.item() if isinstance(d, np.ndarray) else d


out_dir = Path(sys.argv[1])
out_dir.mkdir(parents=True, exist_ok=True)
for p in map(Path, sys.argv[2:]):
    d = load(p)
    pos = np.asarray(d["trajectories"], dtype=float)
    with h5py.File(out_dir / (p.stem + ".h5"), "w") as h:
        h.create_dataset("positions", data=pos, compression="gzip", compression_opts=1)
        h.attrs["frames_per_second"] = float(d["frames_per_second"])
        h.attrs["body_length"] = float(d["body_length"])
        h.attrs["source"] = str(p)
        for k, v in (d.get("setup_points") or {}).items():
            h.create_dataset(f"setup_points/{k}", data=np.asarray(v, dtype=float))
    print(p.stem, pos.shape, "fps", d["frames_per_second"], "BL", d["body_length"], "missing",
          f"{np.isnan(pos).any(axis=2).mean():.3%}", "setup points", sorted((d.get("setup_points") or {}).keys()), flush=True)
