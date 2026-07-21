#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"
DATASET_ROOT="/data2/datasets/seizeit2-dataset/raw-data/ds005873-1.1.0"
OUTPUT_DIR="script_output"
PATIENTS=(
    sub-001 sub-002 sub-003 sub-004 sub-005 sub-006 sub-007 sub-008 sub-009 sub-010
    sub-011 sub-012 sub-013 sub-014 sub-015 sub-016 sub-017 sub-018 sub-019 sub-020
    sub-021 sub-022 sub-023 sub-024 sub-025 sub-026 sub-027 sub-028 sub-029 sub-030
    sub-031 sub-032 sub-033 sub-034 sub-035 sub-036 sub-037 sub-038 sub-039 sub-040
    sub-041 sub-042 sub-043 sub-044 sub-045 sub-046 sub-047 sub-048 sub-049 sub-050
    sub-051 sub-052 sub-053 sub-054 sub-055 sub-056 sub-057 sub-058 sub-059 sub-060
    sub-061 sub-062 sub-063 sub-064 sub-065 sub-066 sub-067 sub-068 sub-069 sub-070
    sub-071 sub-072 sub-073 sub-074 sub-075 sub-076 sub-077 sub-078 sub-079 sub-080
    sub-081 sub-082 sub-083 sub-084 sub-085 sub-086 sub-087 sub-088 sub-089 sub-090
    sub-091 sub-092 sub-093 sub-094 sub-095 sub-096 sub-097 sub-098 sub-099 sub-100
    sub-101 sub-102 sub-103 sub-104 sub-105 sub-106 sub-107 sub-108 sub-109 sub-110
    sub-111 sub-112 sub-113 sub-114 sub-115 sub-116 sub-117 sub-118 sub-119 sub-120
    sub-121 sub-122 sub-123 sub-124 sub-125
)

usage() {
    echo "Usage: $0 [-d|--dataset-root CHEMIN] [-p|--patients sub-001,sub-002,...] [-o|--output-dir CHEMIN]"
    echo "  -d, --dataset-root  Racine du dataset SeizeIT2 (defaut: $DATASET_ROOT)"
    echo "  -p, --patients      Liste de sous-dossiers sub-XXX separes par des virgules (defaut: ${PATIENTS[*]})"
    echo "  -o, --output-dir    Repertoire de sortie pour les fichiers generes (defaut: $OUTPUT_DIR)"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--dataset-root)
            DATASET_ROOT="$2"
            shift 2
            ;;
        -p|--patients)
            IFS=',' read -r -a PATIENTS <<< "$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Option inconnue : $1" >&2
            usage
            exit 1
            ;;
    esac
done

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
    if "$VENV_PYTHON" "$SCRIPT_DIR/edf_to_features.py" --file "$ecg_file" --output-dir "$OUTPUT_DIR"; then
        ok=$(( ok + 1 ))
    else
        echo "ERREUR sur : $ecg_file" >&2
        fail=$(( fail + 1 ))
    fi
done

echo ""
echo "Terminé : $ok succès, $fail échec(s)."
