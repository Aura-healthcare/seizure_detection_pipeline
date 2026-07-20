"""
Tests for main.py.

Only the --rr-file path is tested for now: the QRS detection path from a CSV
fails on pandas 3.x due to an incompatible timestamp format in main.py:106.

Non-regression: compares the produced features against the reference file
tests/data/reference-main-hrv.csv committed in the repo.

Smoke: verifies that the features output file exists and is non-empty.
"""
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "main.py"

# Always use the .venv Python so project dependencies are available regardless
# of which python/pytest the user invoked.
_venv_python = REPO_ROOT / ".venv" / "bin" / "python"
PYTHON = str(_venv_python) if _venv_python.exists() else sys.executable

IGNORE_COLS = {"filename", "original_filename"}
ATOL = 1e-8
RTOL = 1e-5

SAMPLE_CSV = REPO_ROOT / "example_samples" / "sub-001_ses-001_run-01_sample100000.csv"
SAMPLE_RR = (
    REPO_ROOT
    / "example_samples_output"
    / "edf_origin"
    / "sub-001_ses-001_run-01_sample100000"
    / "fast"
    / "rr_sub-001_ses-001_run-01_sample100000_fast.csv"
)
REFERENCE_HRV = REPO_ROOT / "tests" / "data" / "reference-main-hrv.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_main_rr_file(output_dir: pathlib.Path) -> pathlib.Path:
    """Run main.py with --rr-file and return the path of the features file."""
    result = subprocess.run(
        [
            PYTHON, str(SCRIPT),
            "--algo", "fast",
            "--file", str(SAMPLE_CSV),
            "--rr-file", str(SAMPLE_RR),
            "--output-dir", str(output_dir),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, (
        f"main.py failed (exit {result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    stem = SAMPLE_CSV.stem
    return output_dir / stem / "fast" / "features" / f"feats_{stem}_fast.csv"


def _compare_dataframes(actual: pd.DataFrame, expected: pd.DataFrame, label: str) -> None:
    """Compare two DataFrames value-by-value with numeric tolerance."""
    assert actual.shape[0] == expected.shape[0], (
        f"{label}: row count mismatch "
        f"(actual={actual.shape[0]}, expected={expected.shape[0]})"
    )

    compare_cols = [c for c in expected.columns if c not in IGNORE_COLS and c in actual.columns]
    missing = [c for c in expected.columns if c not in IGNORE_COLS and c not in actual.columns]
    assert not missing, f"{label}: columns missing from output: {missing}"

    errors = []
    for col in compare_cols:
        s_act = actual[col].reset_index(drop=True)
        s_exp = expected[col].reset_index(drop=True)

        if pd.api.types.is_numeric_dtype(s_exp):
            a = s_act.to_numpy(dtype=float, na_value=np.nan)
            e = s_exp.to_numpy(dtype=float, na_value=np.nan)
            both_nan = np.isnan(a) & np.isnan(e)
            close = np.isclose(a, e, atol=ATOL, rtol=RTOL, equal_nan=False)
            bad = ~(both_nan | close)
            if bad.any():
                idxs = np.where(bad)[0][:5]
                details = "; ".join(f"row {i}: actual={a[i]!r} expected={e[i]!r}" for i in idxs)
                errors.append(f"  col '{col}': {bad.sum()} mismatch(es) — {details}")
        else:
            mismatch = s_act.astype(str) != s_exp.astype(str)
            if mismatch.any():
                idxs = mismatch[mismatch].index[:5]
                details = "; ".join(
                    f"row {i}: actual={s_act[i]!r} expected={s_exp[i]!r}" for i in idxs
                )
                errors.append(f"  col '{col}': {mismatch.sum()} mismatch(es) — {details}")

    assert not errors, f"{label}: differences found:\n" + "\n".join(errors)


# ---------------------------------------------------------------------------
# Session fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def main_hrv_output(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("main_out")
    return run_main_rr_file(out_dir)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

class TestSmoke:
    def test_hrv_file_exists_and_nonempty(self, main_hrv_output):
        assert main_hrv_output.exists(), f"Features file missing: {main_hrv_output}"
        df = pd.read_csv(main_hrv_output)
        assert len(df) > 0, "Features file contains no data rows"


# ---------------------------------------------------------------------------
# Non-regression tests
# ---------------------------------------------------------------------------

class TestNonRegression:
    def test_hrv_matches_reference(self, main_hrv_output):
        actual = pd.read_csv(main_hrv_output)
        expected = pd.read_csv(REFERENCE_HRV)
        _compare_dataframes(actual, expected, "HRV features (main.py --rr-file)")
