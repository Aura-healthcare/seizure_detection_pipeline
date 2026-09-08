#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser("Split seizure-annotations into one CSV per patient-id")
    p.add_argument("--input", required=True, help="Path to seizure-annotations-v1.0.csv")
    p.add_argument("--outdir", required=True, help="Output directory (will be created)")
    p.add_argument("--patient-col", default="patient-id", help="Column name for patient id")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    if args.patient_col not in df.columns:
        raise ValueError(
            f"Column '{args.patient_col}' not found. Available columns: {list(df.columns)}"
        )

    # Keep original order within each patient (optional)
    for pid, g in df.groupby(args.patient_col, sort=False):
        # Clean filename-safe patient id
        safe_pid = str(pid).replace("/", "_")
        out_path = outdir / f"seizure-annotations_{safe_pid}.csv"
        g.to_csv(out_path, index=False)

    print(f"Done. Wrote {df[args.patient_col].nunique()} files to {outdir}")


if __name__ == "__main__":
    main()