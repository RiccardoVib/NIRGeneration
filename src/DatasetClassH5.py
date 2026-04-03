import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import math

class H5Dataset(Dataset):
    """
    Reads full recordings from an HDF5 (built by build_hdf5/write_hdf5),
    slices them into overlapping segments, caches a segmented HDF5,
    then serves segments via __getitem__.

    HDF5 input layout:
        /attrs          num_samples
        /data/{i}/      input (T,), output (T,), cond (C,)

    Segmented HDF5 layout (auto-generated):
        /attrs          num_segments, seg_len, overlap, hop
        /segments/{i}/  input (seg_len,), output (seg_len,), cond (C,)

    Args:
        source_h5     : path to the original full-recording HDF5
        seg_len       : segment length in samples
        overlap       : overlap between consecutive segments in samples
        force_rebuild : if True, always rebuild the segmented HDF5
    """

    def __init__(
        self,
        source_h5: str,
        seg_len: int,
        overlap: int,
        force_rebuild: bool = False,
    ):
        assert overlap < seg_len, "overlap must be strictly less than seg_len"

        self.source_h5 = Path(source_h5)
        self.seg_len   = seg_len
        self.overlap   = overlap
        self.hop       = seg_len - overlap
        self._file     = None

        # Segmented cache lives alongside source
        stem = self.source_h5.stem
        self.seg_h5_path = self.source_h5.with_name(
            f"{stem}_seg{seg_len}_ov{overlap}.h5"
        )

        if force_rebuild or not self.seg_h5_path.exists():
            self._build_segments()

        with h5py.File(self.seg_h5_path, "r") as f:
            self.length = f.attrs["num_segments"]

    # ── Build ──────────────────────────────────────────────────────────────

    def _build_segments(self):
        print(f"Building segmented HDF5 → {self.seg_h5_path}")
        seg_idx = 0

        with h5py.File(self.source_h5, "r") as src, \
             h5py.File(self.seg_h5_path, "w") as dst:

            num_samples = src.attrs["num_samples"]
            grp = dst.create_group("segments")

            for i in tqdm(range(num_samples), desc="Segmenting"):
                inp  = src[f"data/{i}/input"][:]   # (T,)
                out  = src[f"data/{i}/output"][:]  # (T,)
                cond = src[f"data/{i}/cond"][:]    # (C,)

                T = min(len(inp), len(out))
                n_segs = math.floor((T - self.overlap) / self.hop)

                for s in range(n_segs):
                    start = s * self.hop
                    end   = start + self.seg_len
                    if end > T:
                        break

                    g = grp.create_group(str(seg_idx))
                    g.create_dataset("input",  data=inp[start:end],
                                     compression="gzip", compression_opts=4)
                    g.create_dataset("output", data=out[start:end],
                                     compression="gzip", compression_opts=4)
                    g.create_dataset("cond",   data=cond)
                    seg_idx += 1

            dst.attrs["num_segments"] = seg_idx
            dst.attrs["seg_len"]      = self.seg_len
            dst.attrs["overlap"]      = self.overlap
            dst.attrs["hop"]          = self.hop

        print(f"Done — {seg_idx} segments saved.")

    # ── Lazy file handle (safe for DataLoader workers) ─────────────────────

    def _get_handle(self):
        if self._file is None:
            self._file = h5py.File(self.seg_h5_path, "r", swmr=True)
        return self._file

    # ── Dataset interface ──────────────────────────────────────────────────

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        f    = self._get_handle()
        grp  = f[f"segments/{idx}"]
        inp  = torch.from_numpy(grp["input"][:]).unsqueeze(-1)
        out  = torch.from_numpy(grp["output"][:]).unsqueeze(-1)
        cond = torch.from_numpy(grp["cond"][:]).unsqueeze(0)
        return (inp, cond), out

    def __del__(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
