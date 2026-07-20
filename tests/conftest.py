import pathlib
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def sample_edf() -> pathlib.Path:
    p = REPO_ROOT / "example_samples" / "sub-001_ses-001_run-01_sample100000.edf"
    assert p.exists(), f"Sample EDF not found: {p}"
    return p


@pytest.fixture(scope="session")
def reference_rr() -> pathlib.Path:
    return REPO_ROOT / "tests" / "data" / "reference-rr-interval.csv"


@pytest.fixture(scope="session")
def reference_hrv() -> pathlib.Path:
    return REPO_ROOT / "tests" / "data" / "reference-hrv.csv"
