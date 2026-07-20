"""
Tests for edf_to_features.py.

Non-regression: compares outputs value-by-value against reference files committed
in tests/data/ (rtol=1e-5, atol=1e-8, NaN handled correctly).

Smoke: verifies that the RR-interval and features output files exist and are non-empty.
"""
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "edf_to_features.py"

# Always use the .venv Python so project dependencies are available regardless
# of which python/pytest the user invoked.
_venv_python = REPO_ROOT / ".venv" / "bin" / "python"
PYTHON = str(_venv_python) if _venv_python.exists() else sys.executable

IGNORE_COLS = {"filename", "original_filename"}
ATOL = 1e-8
RTOL = 1e-5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_pipeline(edf_path: pathlib.Path, output_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """Run edf_to_features.py and return (rr_path, hrv_path)."""
    result = subprocess.run(
        [PYTHON, str(SCRIPT), "--file", str(edf_path), "--output-dir", str(output_dir)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, (
        f"edf_to_features.py failed (exit {result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )

    stem = edf_path.stem
    rr_path = output_dir / stem / "fast" / f"rr_{stem}_fast.csv"
    hrv_path = output_dir / stem / "fast" / "features" / f"feats_{stem}_fast.csv"
    return rr_path, hrv_path


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
# Session fixture: pipeline runs once for all tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pipeline_outputs(tmp_path_factory, sample_edf):
    out_dir = tmp_path_factory.mktemp("pipeline_out")
    rr_path, hrv_path = run_pipeline(sample_edf, out_dir)
    return rr_path, hrv_path


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

class TestSmoke:
    def test_rr_file_exists_and_nonempty(self, pipeline_outputs):
        rr_path, _ = pipeline_outputs
        assert rr_path.exists(), f"RR file missing: {rr_path}"
        df = pd.read_csv(rr_path)
        assert len(df) > 0, "RR file contains no data rows"

    def test_hrv_file_exists_and_nonempty(self, pipeline_outputs):
        _, hrv_path = pipeline_outputs
        assert hrv_path.exists(), f"Features file missing: {hrv_path}"
        df = pd.read_csv(hrv_path)
        assert len(df) > 0, "Features file contains no data rows"


# ---------------------------------------------------------------------------
# Non-regression tests
# ---------------------------------------------------------------------------

class TestNonRegression:
    def test_rr_matches_reference(self, pipeline_outputs, reference_rr):
        rr_path, _ = pipeline_outputs
        actual = pd.read_csv(rr_path)
        expected = pd.read_csv(reference_rr)
        _compare_dataframes(actual, expected, "RR intervals")

    def test_hrv_matches_reference(self, pipeline_outputs, reference_hrv):
        _, hrv_path = pipeline_outputs
        actual = pd.read_csv(hrv_path)
        expected = pd.read_csv(reference_hrv)
        _compare_dataframes(actual, expected, "HRV features")
