"""Shared numeric CSV loading for checking, smoke tests and fitting.

Accept headers or headerless data, preserving a missing first measurement.
"""
import csv
from pathlib import Path
import numpy as np


def load_dataset(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.reader(handle) if row]
    if not rows:
        return np.empty((0, 0))
    # Only a nonnumeric, nonempty time label identifies a header. Bad cells in
    # a numeric data row must not cause that entire row to be silently dropped.
    try:
        float(rows[0][0] or "nan")
    except ValueError:
        rows = rows[1:]
    if not rows:
        return np.empty((0, 0))
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("Dataset rows have inconsistent column counts")
    return np.array([[float(cell.strip() or "nan") for cell in row] for row in rows], dtype=float)
