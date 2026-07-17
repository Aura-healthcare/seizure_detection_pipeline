"""
Tests pour edf_to_features.py.

Niveau 1 – non-régression : compare valeur à valeur les sorties avec des fichiers
de référence committés dans tests/data/.

Niveau 2 – smoke test : vérifie que les fichiers de sortie RR-interval et features
sont non vides (plus d'une ligne de header).
"""
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "edf_to_features.py"

# Utilise toujours le Python du .venv pour que les dépendances projet soient disponibles,
# peu importe quel python/pytest a été invoqué par l'utilisateur.
_venv_python = REPO_ROOT / ".venv" / "bin" / "python"
PYTHON = str(_venv_python) if _venv_python.exists() else sys.executable

IGNORE_COLS = {"filename", "original_filename"}
ATOL = 1e-8
RTOL = 1e-5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_pipeline(edf_path: pathlib.Path, output_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """Exécute edf_to_features.py et retourne (rr_path, hrv_path)."""
    result = subprocess.run(
        [PYTHON, str(SCRIPT), "--file", str(edf_path), "--output-dir", str(output_dir)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, (
        f"edf_to_features.py a échoué (exit {result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )

    stem = edf_path.stem
    rr_path = output_dir / stem / "fast" / f"rr_{stem}_fast.csv"
    hrv_path = output_dir / stem / "fast" / "features" / f"feats_{stem}_fast.csv"
    return rr_path, hrv_path


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
# Fixture session : pipeline exécuté une seule fois pour tous les tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pipeline_outputs(tmp_path_factory, sample_edf):
    out_dir = tmp_path_factory.mktemp("pipeline_out")
    rr_path, hrv_path = run_pipeline(sample_edf, out_dir)
    return rr_path, hrv_path


# ---------------------------------------------------------------------------
# Niveau 2 — Smoke tests
# ---------------------------------------------------------------------------

class TestSmoke:
    def test_rr_file_exists_and_nonempty(self, pipeline_outputs):
        rr_path, _ = pipeline_outputs
        assert rr_path.exists(), f"Fichier RR absent : {rr_path}"
        df = pd.read_csv(rr_path)
        assert len(df) > 0, "Le fichier RR ne contient aucune ligne de données"

    def test_hrv_file_exists_and_nonempty(self, pipeline_outputs):
        _, hrv_path = pipeline_outputs
        assert hrv_path.exists(), f"Fichier features absent : {hrv_path}"
        df = pd.read_csv(hrv_path)
        assert len(df) > 0, "Le fichier features ne contient aucune ligne de données"


# ---------------------------------------------------------------------------
# Niveau 1 — Tests de non-régression
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
