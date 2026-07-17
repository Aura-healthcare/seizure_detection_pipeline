"""
Tests pour main.py.

Seul le chemin --rr-file est testé pour l'instant : le chemin de détection QRS
depuis un CSV plante sur pandas 3.x (format timestamp incompatible).

Niveau 1 – non-régression : compare les features produites avec le fichier de
référence tests/data/reference-main-hrv.csv commité dans le repo.

Niveau 2 – smoke test : vérifie que le fichier features produit est non vide.
"""
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "main.py"

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
    """Exécute main.py avec --rr-file et retourne le chemin du fichier features."""
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
        f"main.py a échoué (exit {result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    stem = SAMPLE_CSV.stem
    return output_dir / stem / "fast" / "features" / f"feats_{stem}_fast.csv"


def _compare_dataframes(actual: pd.DataFrame, expected: pd.DataFrame, label: str) -> None:
    """Compare deux DataFrames valeur à valeur avec tolérance numérique."""
    assert actual.shape[0] == expected.shape[0], (
        f"{label}: nombre de lignes différent "
        f"(actual={actual.shape[0]}, expected={expected.shape[0]})"
    )

    compare_cols = [c for c in expected.columns if c not in IGNORE_COLS and c in actual.columns]
    missing = [c for c in expected.columns if c not in IGNORE_COLS and c not in actual.columns]
    assert not missing, f"{label}: colonnes manquantes dans la sortie : {missing}"

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

    assert not errors, f"{label}: différences détectées :\n" + "\n".join(errors)


# ---------------------------------------------------------------------------
# Fixture session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def main_hrv_output(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("main_out")
    return run_main_rr_file(out_dir)


# ---------------------------------------------------------------------------
# Niveau 2 — Smoke test
# ---------------------------------------------------------------------------

class TestSmoke:
    def test_hrv_file_exists_and_nonempty(self, main_hrv_output):
        assert main_hrv_output.exists(), f"Fichier features absent : {main_hrv_output}"
        df = pd.read_csv(main_hrv_output)
        assert len(df) > 0, "Le fichier features ne contient aucune ligne de données"


# ---------------------------------------------------------------------------
# Niveau 1 — Test de non-régression
# ---------------------------------------------------------------------------

class TestNonRegression:
    def test_hrv_matches_reference(self, main_hrv_output):
        actual = pd.read_csv(main_hrv_output)
        expected = pd.read_csv(REFERENCE_HRV)
        _compare_dataframes(actual, expected, "HRV features (main.py --rr-file)")
