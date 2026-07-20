#!/usr/bin/env python3
"""
Merge feat-hrv, feat-acc, and seizure-annotation into a unified feat-grid.csv.

Usage:
    python merge_feat_grid.py \
        --hrv   path/to/feat-hrv.csv \
        --acc   path/to/feat-acc.csv \
        --seizure path/to/seizure-annotation.csv \
        --output  path/to/feat-grid.csv
"""

import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge HRV, ACC, and seizure annotation features into a unified time-grid CSV."
    )
    parser.add_argument("--hrv",            required=True,  help="Path to feat-hrv CSV")
    parser.add_argument("--acc",            required=True,  help="Path to feat-acc CSV")
    parser.add_argument("--seizure",        required=True,  help="Path to seizure-annotation CSV")
    parser.add_argument("--output-union",    required=True,  help="Output path for feat-grid-union.csv (hrv OR acc available)")
    parser.add_argument("--output-intersect",required=True,  help="Output path for feat-grid-intersection.csv (hrv AND acc available)")
    parser.add_argument("--training-split", required=True,  help="Training split label (e.g. train, val, test)")
    parser.add_argument("--patient-id",     required=True,  help="Patient identifier (e.g. 01-001)")
    return parser.parse_args()


def load_hrv(path: str) -> pd.DataFrame:
    print(f"  Loading HRV from {path} ...")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.drop(columns=["filename", "original_filename"], errors="ignore")
    df.columns = [f"hrv_{c}" if c != "timestamp" else c for c in df.columns]
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_acc(path: str) -> pd.DataFrame:
    print(f"  Loading ACC from {path} ...")
    df = pd.read_csv(path)

    # Drop unnamed row-index column if present
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    df = df.drop(columns=unnamed, errors="ignore")

    # Normalise ACC timestamp format: "2026-03-12_14:15:00.359000+0100"
    # → "2026-03-12 14:15:00.359000+01:00"
    df["time"] = (
        df["time"]
        .str.replace("_", " ", regex=False)
        .str.replace(r"\+(\d{2})(\d{2})$", r"+\1:\2", regex=True)
    )
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df.columns = [f"acc_{c}" if c != "time" else c for c in df.columns]
    df = df.sort_values("time").reset_index(drop=True)
    return df


def load_seizures(path: str) -> pd.DataFrame:
    print(f"  Loading seizure annotations from {path} ...")
    df = pd.read_csv(path)
    for col in ["reference-seizure-start-date", "reference-seizure-end-date",
                "aura-seizure-start-date", "aura-seizure-end-date"]:
        df[col] = pd.to_datetime(df[col], utc=True)
    return df


def annotate_seizures(grid: pd.DataFrame, seizures: pd.DataFrame) -> pd.DataFrame:
    grid["label"] = 0
    grid["seizure-id"] = pd.NA

    for _, row in seizures.iterrows():
        mask = (grid.index >= row["aura-seizure-start-date"]) & \
               (grid.index <= row["aura-seizure-end-date"])
        grid.loc[mask, "label"] = 1
        grid.loc[mask, "seizure-id"] = row["seizure-id"]

    return grid


def main():
    args = parse_args()

    print("── Loading files ──────────────────────────────────────────────")
    hrv      = load_hrv(args.hrv)
    acc      = load_acc(args.acc)
    seizures = load_seizures(args.seizure)

    # ── Build 1-second time grid spanning both signals ────────────────────────
    t_min = min(hrv["timestamp"].min(), acc["time"].min())
    t_max = max(hrv["timestamp"].max(), acc["time"].max())
    print(f"\n── Building time grid ─────────────────────────────────────────")
    print(f"  From : {t_min}")
    print(f"  To   : {t_max}")

    grid = pd.DataFrame(index=pd.date_range(start=t_min, end=t_max, freq="1s", name="timestamps"))
    print(f"  Rows : {len(grid):,}")

    # ── Left-join HRV and ACC onto the grid ───────────────────────────────────
    print("\n── Merging ────────────────────────────────────────────────────")
    print("  Joining HRV  (nearest match ±500 ms) ...")
    grid_reset = grid.reset_index()
    merged = pd.merge_asof(
        grid_reset,
        hrv,
        left_on="timestamps",
        right_on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("500ms"),
    )
    merged["hrv_available"] = merged["timestamp"].notna()
    merged = merged.drop(columns=["timestamp"]).set_index("timestamps")
    grid = merged

    print("  Joining ACC  (nearest match ±2500 ms) ...")
    grid_reset = grid.reset_index()
    merged_acc = pd.merge_asof(
        grid_reset,
        acc,
        left_on="timestamps",
        right_on="time",
        direction="nearest",
        tolerance=pd.Timedelta("2500ms"),
    )
    merged_acc["acc_available"] = merged_acc["time"].notna()
    merged_acc = merged_acc.drop(columns=["time"]).set_index("timestamps")
    grid = merged_acc

    # ── Seizure flags ─────────────────────────────────────────────────────────
    print("  Annotating seizure windows ...")
    grid = annotate_seizures(grid, seizures)

    # ── Metadata columns ──────────────────────────────────────────────────────
    grid["training-split"] = args.training_split
    grid["patient-id"]     = args.patient_id

    # ── Filter and write outputs ───────────────────────────────────────────────
    grid.index = grid.index.tz_convert("Europe/Paris")

    grid_union  = grid[grid["hrv_available"] | grid["acc_available"]]
    grid_inters = grid[grid["hrv_available"] & grid["acc_available"]]

    print(f"\n── Writing outputs ────────────────────────────────────────────")
    print(f"  Union        : {len(grid_union):,} rows  (hrv OR  acc available) → {args.output_union}")
    grid_union.to_csv(args.output_union)

    print(f"  Intersection : {len(grid_inters):,} rows (hrv AND acc available) → {args.output_intersect}")
    grid_inters.to_csv(args.output_intersect)

    print("  Done.")


if __name__ == "__main__":
    main()
