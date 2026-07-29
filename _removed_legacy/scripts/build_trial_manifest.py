"""Rebuild data/processed/trial_manifest.csv by scanning every .mat file under
data/raw/{NLoS,PLoS}/csi_office_*ms_interframe.mat.

Only reads HDF5 metadata (via mat_loader.peek_csi_shape), never loads a full
CSI matrix, so this is cheap even against multi-hundred-MB files.

Run:
    .venv/bin/python scripts/build_trial_manifest.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from snn_csi_tracking.data import mat_loader

DATA_ROOT = Path(__file__).parent.parent / "data" / "raw"
OUT_PATH = Path(__file__).parent.parent / "data" / "processed" / "trial_manifest.csv"

LABEL_TO_ID = {code: i for i, code in enumerate(mat_loader.ACTIVITY_CODES)}

FIELDNAMES = [
    "condition",
    "interframe_ms",
    "filename",
    "room",
    "activity_code",
    "label_id",
    "activity",
    "num_csi",
    "num_subcarriers",
    "num_antennas",
    "csi_available",
]


def main():
    records = mat_loader.discover_records(DATA_ROOT)
    out_rows = []
    for record in records:
        rows = mat_loader.load_table_metadata(record.path)
        for row in rows:
            shape = mat_loader.peek_csi_shape(record.path, row["csi_key"], row["row_index"])
            available = shape is not None
            num_csi, num_subcarriers, num_antennas = shape if available else (None, None, None)
            out_rows.append(
                {
                    "condition": record.condition,
                    "interframe_ms": record.interframe_ms,
                    "filename": row["filename"],
                    "room": row["room"],
                    "activity_code": row["activity_code"],
                    "label_id": LABEL_TO_ID.get(row["activity_code"]),
                    "activity": row["activity"],
                    "num_csi": num_csi,
                    "num_subcarriers": num_subcarriers,
                    "num_antennas": num_antennas,
                    "csi_available": available,
                }
            )
            print(
                f"{record.condition}/{record.interframe_ms}ms/{row['filename']}/"
                f"{row['activity_code']}: shape={shape}"
            )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nwrote {len(out_rows)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
