"""Loading of the raw CSI .mat structures.

Confirmed from the raw files (2026-07-08):

    data/raw/
        NLoS/
            csi_office_5ms_interframe.mat     (~13.7 GB)
            csi_office_10ms_interframe.mat    (~6.3 GB)
            csi_office_50ms_interframe.mat    (~1.3 GB)
            csi_office_100ms_interframe.mat   (~12 MB)
        PLoS/
            csi_office_5ms_interframe.mat     (~9.9 GB)
            csi_office_10ms_interframe.mat    (~6.6 GB)
            csi_office_50ms_interframe.mat    (~1.3 GB)
            csi_office_100ms_interframe.mat   (~14 MB)

The "interframe" value is the sampling interval, i.e. frame rate = 1 / interval.
All 8 files are MATLAB v7.3 (HDF5-based), including the 100ms ones.

SCHEMA: every file stores one MATLAB `table` with columns
[FileName, FullPath, Room, ActivityCode, Activity, fullCSI] and ~25 rows (one
row per activity x capture-repeat trial; some files have fewer rows because a
capture failed during collection and was dropped entirely -- row counts
observed range from 18 to 25). Columns:

    FileName      e.g. "capture1.mat"          -- which repeat (1..5) of the trial
    FullPath      original path on the collector's machine -- provenance only
    Room          constant "office" in every row -- not a useful feature
    ActivityCode  one of {"E", "EN-S", "EN-W", "L", "S"}   <- classification LABEL
    Activity      human description (see ACTIVITY_DESCRIPTIONS below)
    fullCSI       the CSI matrix for that trial (complex128)

CSI array shape is NOT consistent across interframe rates:
    5ms / 10ms / 50ms files: raw shape (NumAntennas=3, NumSubcarriers=1024, NumCSI)
    100ms files:             raw shape (NumSubcarriers=64, NumCSI)  -- no antenna axis

load_csi() below always returns the (NumCSI, NumSubcarriers, NumAntennas)
convention, inserting NumAntennas=1 for the 100ms files. NumCSI (trial
duration in frames) varies per trial, so CSI arrays must be windowed (see
preprocessing.sliding_windows) rather than stacked directly.

A single 5ms-interframe trial can be ~500 MB as complex128 (3 x 1024 x ~12000
x 16 bytes), so CSI matrices are NOT eagerly loaded -- load_table_metadata()
returns lightweight row dicts, and load_csi() fetches one trial's matrix on
demand. Occasionally a row's fullCSI cell is a placeholder rather than a real
matrix (a capture that failed during collection); load_csi() returns None for
those rows and callers should skip them.

MATLAB `table` objects are stored in v7.3 .mat files via an undocumented,
version-specific binary format (MCOS/"FileWrapper__") that no Python library
(scipy, h5py, pymatreader, mat73) decodes -- all four were tried against
these files and either raised or silently returned raw uint32 reference
arrays instead of table contents. The functions below parse the format
directly, based on structure reverse-engineered empirically against these
8 files (see PR description for the byte-level walkthrough).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CONDITIONS = ("NLoS", "PLoS")
FRAME_INTERVALS_MS = (5, 10, 50, 100)

ACTIVITY_CODES = ("E", "EN-S", "EN-W", "L", "S")
ACTIVITY_DESCRIPTIONS = {
    "E": "empty room",
    "EN-S": "person enters then sits",
    "EN-W": "person enters then walks",
    "L": "person sits then stands and leaves",
    "S": "person sitting",
}

# Documents the table's column -> semantic role. Column values are matched by
# content (see load_table_metadata) rather than by this map directly, since
# each column needs type-specific binary decoding.
DEFAULT_FIELD_MAP = {
    "csi": "fullCSI",
    "label": "ActivityCode",
    "label_description": "Activity",
}


@dataclass(frozen=True)
class Record:
    path: Path
    condition: str
    interframe_ms: int


def discover_records(root: str | Path, conditions: tuple[str, ...] = CONDITIONS) -> list[Record]:
    """Walk `root/{condition}/csi_office_<N>ms_interframe.mat` and list what's on disk."""
    root = Path(root)
    records: list[Record] = []
    for condition in conditions:
        condition_dir = root / condition
        if not condition_dir.is_dir():
            continue
        for interval in FRAME_INTERVALS_MS:
            path = condition_dir / f"csi_office_{interval}ms_interframe.mat"
            if path.exists():
                records.append(Record(path=path, condition=condition, interframe_ms=interval))
    return records


def open_mat(path: str | Path) -> tuple[Any, str]:
    """Open a .mat file, transparently handling both classic and v7.3 (HDF5) formats.

    Returns (handle, backend) where backend is "scipy" or "h5py". For the
    "h5py" backend the caller is responsible for closing the returned handle.
    All 8 csi_office_*.mat files are v7.3, so backend will be "h5py" for them.
    """
    import scipy.io

    try:
        return scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False), "scipy"
    except NotImplementedError:
        import h5py

        return h5py.File(path, "r"), "h5py"


def inspect_mat(path: str | Path) -> dict:
    """Report the top-level keys (and shapes/dtypes where cheap to get) of a .mat file.

    For the csi_office_*.mat files this only shows raw HDF5 dataset paths
    (#refs#/..., #subsystem#/...), not the table's logical columns -- use
    load_table_metadata() for that.
    """
    handle, backend = open_mat(path)
    try:
        if backend == "scipy":
            return {
                k: getattr(v, "shape", type(v).__name__)
                for k, v in handle.items()
                if not k.startswith("__")
            }
        else:
            info = {}

            def _visit(name, obj):
                import h5py as _h5py

                if isinstance(obj, _h5py.Dataset):
                    info[name] = (obj.shape, str(obj.dtype))

            handle.visititems(_visit)
            return info
    finally:
        if backend == "h5py":
            handle.close()


def _try_decode_ascii(arr: np.ndarray) -> str | None:
    arr = np.asarray(arr).ravel()
    if arr.dtype.kind not in ("u", "i") or arr.size == 0:
        return None
    if arr.max(initial=0) >= 0x110000:
        return None
    chars = "".join(chr(int(x)) for x in arr if x != 0)
    if chars and all(32 <= ord(c) < 127 for c in chars):
        return chars
    return None


def _decode_string_column(refs, key: str, n_rows: int) -> list[str] | None:
    """Decode a MATLAB char-array column stored as MCOS "compact" uint64 data.

    Layout: [1, 2, n_rows, 1] header, then n_rows per-row string lengths (in
    UTF-16 code units), then all row characters packed back-to-back as
    little-endian UTF-16 across the remaining uint64 words. Returns None if
    `key` doesn't match this layout (i.e. it isn't a string column).
    """
    import h5py

    obj = refs[key]
    if not (isinstance(obj, h5py.Dataset) and obj.dtype.kind == "u" and not obj.dtype.names):
        return None
    v = obj[()].ravel().astype(np.uint64)
    if not (len(v) >= 4 and v[0] == 1 and v[1] == 2 and v[2] == n_rows and v[3] == 1):
        return None
    lengths = v[4 : 4 + n_rows]
    packed = v[4 + n_rows :]
    raw = b"".join(struct.pack("<Q", int(x)) for x in packed)
    chars = raw.decode("utf-16-le", errors="replace")
    out, pos = [], 0
    for length in lengths:
        length = int(length)
        out.append(chars[pos : pos + length])
        pos += length
    return out


def _find_varnames_key(f, refs) -> tuple[str, list[str]] | tuple[None, None]:
    import h5py

    for key in refs.keys():
        obj = refs[key]
        if isinstance(obj, h5py.Dataset) and obj.dtype == object and 2 <= obj.size <= 12:
            decoded = []
            for item in obj[()].ravel():
                if not (isinstance(item, h5py.Reference) and item):
                    decoded = None
                    break
                s = _try_decode_ascii(f[item][()])
                if s is None:
                    decoded = None
                    break
                decoded.append(s)
            if decoded and "fullCSI" in decoded:
                return key, decoded
    return None, None


def _find_csi_cell_key(f, refs) -> str | None:
    """Locate the object-array cell holding one CSI-matrix reference per row."""
    import h5py

    for key in refs.keys():
        obj = refs[key]
        if isinstance(obj, h5py.Dataset) and obj.dtype == object and obj.size >= 15:
            flat = obj[()].ravel()
            complex_count = sum(
                1
                for item in flat
                if isinstance(item, h5py.Reference)
                and item
                and f[item].dtype.names
                and "real" in f[item].dtype.names
            )
            if complex_count >= obj.size - 3:
                return key
    return None


def load_table_metadata(path: str | Path) -> list[dict[str, Any]]:
    """Return one dict per experiment trial, without loading CSI matrices.

    Each dict has: row_index, csi_key, filename, fullpath, room,
    activity_code, activity. Pass row_index/csi_key to load_csi() to fetch
    that trial's (potentially huge) CSI matrix on demand.
    """
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as f:
        refs = f["#refs#"]
        varnames_key, varnames = _find_varnames_key(f, refs)
        if varnames is None:
            raise ValueError(f"Could not locate table VariableNames in {path}")
        csi_key = _find_csi_cell_key(f, refs)
        if csi_key is None:
            raise ValueError(f"Could not locate the fullCSI cell array in {path}")
        n_rows = refs[csi_key].shape[0] if refs[csi_key].ndim == 1 else refs[csi_key].size

        columns: dict[str, list[str]] = {}
        for key in refs.keys():
            if key in (csi_key, varnames_key):
                continue
            decoded = _decode_string_column(refs, key, n_rows)
            if decoded is not None:
                columns[key] = decoded

        # Map decoded columns to names by content, not position (row-key
        # letters shift between files depending on serialization order).
        by_name: dict[str, list[str]] = {}
        for vals in columns.values():
            uniq = set(vals)
            if uniq <= {"office"}:
                by_name["Room"] = vals
            elif uniq <= set(ACTIVITY_CODES):
                by_name["ActivityCode"] = vals
            elif uniq <= set(ACTIVITY_DESCRIPTIONS.values()):
                by_name["Activity"] = vals
            elif all(v.startswith("capture") for v in vals):
                by_name["FileName"] = vals
            else:
                by_name["FullPath"] = vals

        rows = []
        for i in range(n_rows):
            rows.append(
                {
                    "row_index": i,
                    "csi_key": csi_key,
                    "filename": by_name.get("FileName", [None] * n_rows)[i],
                    "fullpath": by_name.get("FullPath", [None] * n_rows)[i],
                    "room": by_name.get("Room", [None] * n_rows)[i],
                    "activity_code": by_name.get("ActivityCode", [None] * n_rows)[i],
                    "activity": by_name.get("Activity", [None] * n_rows)[i],
                }
            )
        return rows


def peek_csi_shape(path: str | Path, csi_key: str, row_index: int) -> tuple[int, int, int] | None:
    """Return one trial's (NumCSI, NumSubcarriers, NumAntennas) without reading any CSI data.

    Only touches HDF5 dataset metadata (.shape), so this is cheap even against
    the multi-GB files. Returns None for placeholder (missing-capture) rows.
    """
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as f:
        refs = f["#refs#"]
        ref = refs[csi_key][()].ravel()[row_index]
        if not (isinstance(ref, h5py.Reference) and ref):
            return None
        target = f[ref]
        if not (target.dtype.names and "real" in target.dtype.names):
            return None
        if target.ndim == 3:
            antennas, subcarriers, num_csi = target.shape
        else:
            subcarriers, num_csi = target.shape
            antennas = 1
        return (num_csi, subcarriers, antennas)


def load_csi(path: str | Path, csi_key: str, row_index: int) -> np.ndarray | None:
    """Load one trial's CSI matrix as a (NumCSI, NumSubcarriers, NumAntennas) complex128 array.

    Returns None if this row's fullCSI cell is a placeholder (failed capture).
    """
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as f:
        refs = f["#refs#"]
        ref = refs[csi_key][()].ravel()[row_index]
        if not (isinstance(ref, h5py.Reference) and ref):
            return None
        target = f[ref]
        if not (target.dtype.names and "real" in target.dtype.names):
            return None
        raw = target[()]
        complex_arr = raw["real"] + 1j * raw["imag"]
        if complex_arr.ndim == 3:
            # raw (NumAntennas, NumSubcarriers, NumCSI) -> (NumCSI, NumSubcarriers, NumAntennas)
            return np.transpose(complex_arr, (2, 1, 0))
        # raw (NumSubcarriers, NumCSI) -> (NumCSI, NumSubcarriers, 1)
        return np.transpose(complex_arr, (1, 0))[:, :, np.newaxis]


def load_record(record: Record) -> list[dict[str, Any]]:
    """Load all trials from one .mat file's table, attaching file-level metadata.

    CSI matrices are NOT eagerly loaded (a single 5ms-interframe trial can be
    several hundred MB) -- call load_csi(record.path, row["csi_key"],
    row["row_index"]) to fetch one trial's matrix when needed.
    """
    rows = load_table_metadata(record.path)
    for row in rows:
        row["condition"] = record.condition
        row["interframe_ms"] = record.interframe_ms
        row["path"] = str(record.path)
    return rows
