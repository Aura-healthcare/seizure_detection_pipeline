#!/bin/bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 [--train "PATIENT1 PATIENT2 ..."] [--test "PATIENT1 PATIENT2 ..."]

Options:
  --train PATIENTS   Liste des patients train, séparés par des espaces (entre guillemets)
  --test  PATIENTS   Liste des patients test, séparés par des espaces (entre guillemets)
  -h, --help         Affiche cette aide

Exemple:
  $0 --train "01-001 01-002 01-004" --test "01-003 01-006"

Valeurs par défaut (si --train/--test non précisés):
  train: 01-001 01-002 01-004 01-005 01-011
  test:  01-003 01-006 01-007 01-009
EOF
}

# Valeurs par défaut
train_patients="01-001 01-002 01-004 01-005 01-011"
test_patients="01-003 01-006 01-007 01-009"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --train)
            train_patients="$2"
            shift 2
            ;;
        --test)
            test_patients="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Option inconnue: $1" >&2
            usage
            exit 1
            ;;
    esac
done

echo "=== TRAITEMENT DES PATIENTS TRAIN ==="
for patient in $train_patients; do
    echo "Processing TRAIN patient $patient..."
    python merge_feat_grid.py \
        --hrv data/train/feat_hrv_${patient}.csv \
        --acc data/train/features_${patient}.acc.csv \
        --seizure data/seizure_by_patient_v1.2/seizure-annotations_${patient}.csv \
        --output-union output_v1.2/feat-grid-union_${patient}.csv \
        --output-intersect output_v1.2/feat-grid-intersect_${patient}.csv \
        --training-split train \
        --patient-id $patient
done

echo ""
echo "=== TRAITEMENT DES PATIENTS TEST ==="
for patient in $test_patients; do
    echo "Processing TEST patient $patient..."
    python merge_feat_grid.py \
        --hrv data/test/feat_hrv_${patient}.csv \
        --acc data/test/features_${patient}.acc.csv \
        --seizure data/seizure_by_patient_v1.2/seizure-annotations_${patient}.csv \
        --output-union output_v1.2/feat-grid-union_${patient}.csv \
        --output-intersect output_v1.2/feat-grid-intersect_${patient}.csv \
        --training-split test \
        --patient-id $patient
done

echo ""
echo "=== TRAITEMENT TERMINÉ ==="
echo "Train patients: $train_patients"
echo "Test patients: $test_patients"