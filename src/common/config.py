"""Project-wide paths and constants (owner: Person A).

Every laptop stores the raw data in a different place, so the data root is read
from the environment variable PATHSNAP_DATA_ROOT. It must point at the folder
that contains the partition folders `train/` and `student_test/`.

If the variable is not set we fall back to the location used on the Person A
laptop (repo/data_raw/Student folder), so scripts still run out of the box there.
"""
from __future__ import annotations

import os
from pathlib import Path

# Repo root = two levels above this file (src/common/config.py -> repo/)
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path(
    os.environ.get("PATHSNAP_DATA_ROOT", REPO_ROOT / "data_raw" / "Student folder")
)

# Partitions we are allowed to develop on. The evaluator-test partition is only
# ever pointed to by the API at demo time (Person B's DATA_DIR), never used here.
DEV_PARTITIONS = ("train", "student_test")

RESULTS_DIR = REPO_ROOT / "results"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
CACHE_DIR = REPO_ROOT / "data_cache"   # regenerable candidate/feature tables (git-ignored)

# Metric coordinates: a per-trip local azimuthal-equidistant projection
# (trips are worldwide, so no single CRS fits) - see geo.py.

# Train trips whose .route does not match the GPS track (EDA: median GPS-to-route
# distance 120-140 m vs ~5 m for every other trip). Excluded from model FITTING
# only; listed in report/a_section2_notes.md.
ANOMALOUS_TRIPS = {"00000054", "00000033"}

# Global random seed for every experiment (recorded in result files).
SEED = 42
