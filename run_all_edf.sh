#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"
DATASET_ROOT="/data2/datasets/seizeit2-dataset/ds005873-1.1.0"

PATIENTS=("sub-024" "sub-025")

mapfile -t ecg_files < <(
    for sub in "${PATIENTS[@]}"; do
        find "$DATASET_ROOT/$sub" -path "*/ecg/*.edf" -o -path "*/ecg/*.csv" 2>/dev/null
    done | sort
)

if [[ ${#ecg_files[@]} -eq 0 ]]; then
    echo "Aucun fichier ECG EDF/CSV trouvé sous $DATASET_ROOT"
    exit 1
fi

echo "Traitement de ${#ecg_files[@]} fichier(s) ECG (EDF/CSV)..."

ok=0
fail=0

for ecg_file in "${ecg_files[@]}"; do
    echo ""
    echo "=========================================="
    echo "Fichier : $ecg_file"
    echo "=========================================="
    if "$VENV_PYTHON" "$SCRIPT_DIR/edf_to_features.py" --file "$ecg_file"; then
        ok=$(( ok + 1 ))
    else
        echo "ERREUR sur : $ecg_file" >&2
        fail=$(( fail + 1 ))
    fi
done

echo ""
echo "Terminé : $ok succès, $fail échec(s)."
