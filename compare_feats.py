#!/usr/bin/env python3
"""Compare a feats CSV produced from edf_origin against the one from csv_origin.

Usage:
    python compare_feats.py [file_edf] [file_csv]

If no arguments are given, defaults to the sub-001_ses-001_run-01_sample100000
example under example_samples_output/{edf_origin,csv_origin}.
"""
import sys
import numpy as np
import pandas as pd

DEFAULT_EDF = "example_samples_output/edf_origin/sub-001_ses-001_run-01_sample100000/fast/features/feats_sub-001_ses-001_run-01_sample100000_fast.csv"
DEFAULT_CSV = "example_samples_output/csv_origin/sub-001_ses-001_run-01_sample100000/fast/features/feats_sub-001_ses-001_run-01_sample100000_fast.csv"

# Columns expected to differ because they embed the origin folder name.
IGNORE_COLS = {"filename", "original_filename"}

ATOL = 1e-8
RTOL = 1e-5


def main():
    path_edf = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EDF
    path_csv = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_CSV

    df_edf = pd.read_csv(path_edf)
    df_csv = pd.read_csv(path_csv)

    print(f"edf_origin file: {path_edf} ({df_edf.shape[0]} rows, {df_edf.shape[1]} cols)")
    print(f"csv_origin file: {path_csv} ({df_csv.shape[0]} rows, {df_csv.shape[1]} cols)")

    if list(df_edf.columns) != list(df_csv.columns):
        print("\n/!\\ Column names/order differ:")
        print("  edf_origin:", list(df_edf.columns))
        print("  csv_origin:", list(df_csv.columns))
        common_cols = [c for c in df_edf.columns if c in df_csv.columns]
    else:
        common_cols = list(df_edf.columns)

    if df_edf.shape[0] != df_csv.shape[0]:
        print(f"\n/!\\ Row counts differ: edf_origin={df_edf.shape[0]}, csv_origin={df_csv.shape[0]}")

    n_rows = min(df_edf.shape[0], df_csv.shape[0])
    df_edf = df_edf.iloc[:n_rows].reset_index(drop=True)
    df_csv = df_csv.iloc[:n_rows].reset_index(drop=True)

    compare_cols = [c for c in common_cols if c not in IGNORE_COLS]

    total_diffs = 0
    for col in compare_cols:
        s_edf = df_edf[col]
        s_csv = df_csv[col]

        if pd.api.types.is_numeric_dtype(s_edf) and pd.api.types.is_numeric_dtype(s_csv):
            both_nan = (s_edf.isna() & s_csv.isna()).to_numpy()
            close = np.isclose(s_edf.to_numpy(), s_csv.to_numpy(), atol=ATOL, rtol=RTOL, equal_nan=False)
            mismatch = pd.Series(~(both_nan | close), index=s_edf.index)
        else:
            mismatch = ~(s_edf.astype(str) == s_csv.astype(str))

        n_mismatch = int(mismatch.sum())
        if n_mismatch:
            total_diffs += n_mismatch
            print(f"\nColonne '{col}': {n_mismatch} valeur(s) differente(s)")
            idxs = mismatch[mismatch].index[:10]
            for i in idxs:
                print(f"  ligne {i}: edf_origin={s_edf[i]!r}  csv_origin={s_csv[i]!r}")
            if n_mismatch > 10:
                print(f"  ... et {n_mismatch - 10} de plus")

    print(f"\nColonnes ignorees (attendues differentes): {sorted(IGNORE_COLS & set(common_cols))}")

    if total_diffs == 0:
        print("\nResultat: les deux fichiers sont identiques (hors colonnes de chemin ignorees).")
    else:
        print(f"\nResultat: {total_diffs} difference(s) au total sur {len(compare_cols)} colonnes comparees.")


if __name__ == "__main__":
    main()
