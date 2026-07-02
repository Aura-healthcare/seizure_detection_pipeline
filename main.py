import argparse
import os
import pathlib
import sys
from typing import Iterable, Optional

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

CURRENT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PAN_TOMPKINS_DIR = pathlib.Path(
    "/pipeline-scripts/processing/ecg-to-rr-intervals/pan-tompkins"
)
SEIZURE_DETECTION_PIPELINE_DIR = PAN_TOMPKINS_DIR / "seizure_detection_pipeline"
HRV_ANALYSIS_NA_DIR = SEIZURE_DETECTION_PIPELINE_DIR / "src" / "usecase"
for extra_path in (PAN_TOMPKINS_DIR, SEIZURE_DETECTION_PIPELINE_DIR, HRV_ANALYSIS_NA_DIR):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

from sources.features import compute_features
from sources.fast import qrs_detector as fast_qrs_detector
from sources.hamilton import qrs_detector as hamilton_qrs_detector


DEFAULT_OUTPUT_DIR = "script_output"
DEFAULT_SAMPLING_FREQUENCY = 250


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect QRS peaks, build RR intervals, then compute HRV features."
    )
    parser.add_argument("--algo", choices=["fast", "hamilton"], required=True)
    parser.add_argument("--file", required=True, help="Input ECG CSV file.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sampling-frequency", "--fs", type=int, default=DEFAULT_SAMPLING_FREQUENCY)
    parser.add_argument("--time-column", default=None)
    parser.add_argument("--signal-column", default=None)
    parser.add_argument("--chunk-size", type=int, default=3_000_000)
    parser.add_argument("--max-nan-interpolation", type=int, default=10)
    parser.add_argument("--min-segment-len", type=int, default=50)
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--rr-file", default=None, help="Existing RR CSV. If set, skips QRS detection.")
    return parser.parse_args()


def choose_time_column(columns: Iterable[str], explicit: Optional[str]) -> str:
    columns = list(columns)
    if explicit:
        if explicit not in columns:
            raise ValueError(f"Time column '{explicit}' not found. Available columns: {columns}")
        return explicit

    for candidate in ["time", "timestamp", "datetime", "date"]:
        if candidate in columns:
            return candidate

    raise ValueError(
        "Could not infer the time column. Use --time-column. "
        f"Available columns: {columns}"
    )


def choose_signal_column(df: pd.DataFrame, time_column: str, explicit: Optional[str]) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Signal column '{explicit}' not found. Available columns: {df.columns.tolist()}")
        return explicit

    for candidate in ["ecgpoint", "ecg", "signal", "value"]:
        if candidate in df.columns and candidate != time_column:
            return candidate

    numeric_columns = [
        col
        for col in df.columns
        if col != time_column and pd.api.types.is_numeric_dtype(df[col])
    ]
    if len(numeric_columns) == 1:
        return numeric_columns[0]

    raise ValueError(
        "Could not infer the ECG signal column. Use --signal-column. "
        f"Numeric candidates: {numeric_columns}"
    )


def detect_chunk_qrs(
    chunk: pd.DataFrame,
    algo: str,
    sampling_frequency: int,
    time_column: str,
    signal_column: str,
    max_nan_interpolation: int,
    min_segment_len: int,
) -> pd.Series:
    chunk = chunk[[time_column, signal_column]].copy()
    chunk[time_column] = pd.to_datetime(
        chunk[time_column], format="%Y-%m-%d_%H:%M:%S.%f%z"
    )
    chunk[signal_column] = pd.to_numeric(chunk[signal_column], errors="coerce")

    if algo == "fast":
        chunk_for_fast = chunk.rename(columns={time_column: "time", signal_column: "ecgpoint"})
        return fast_qrs_detector(
            chunk_for_fast,
            sampling_frequency,
            max_nan_interpolation=max_nan_interpolation,
            min_segment_len=min_segment_len,
        )

    return hamilton_qrs_detector(
        chunk,
        sampling_frequency,
        time_column=time_column,
        signal_column=signal_column,
    )


def build_rr_dataframe(qrs_detections: list[pd.Series], sampling_frequency: int) -> pd.DataFrame:
    if not qrs_detections:
        raise ValueError("No QRS detections were produced.")

    qrs = pd.concat(qrs_detections)
    if len(qrs) == 0:
        raise ValueError("No QRS peaks detected.")

    df = pd.DataFrame({"timestamp": pd.to_datetime(qrs)})
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    df["rr_interval"] = df["timestamp"].diff().dt.total_seconds() * 1000
    df["rr_interval"] = df["rr_interval"].fillna(0)

    start_time = df["timestamp"].iloc[0]
    df["frame"] = ((df["timestamp"] - start_time).dt.total_seconds() * sampling_frequency).round().astype(int)
    df["time"] = df["timestamp"]
    return df[["timestamp", "frame", "rr_interval", "time"]]


def output_paths(input_file: str, output_dir: str, algo: str) -> tuple[pathlib.Path, pathlib.Path]:
    input_path = pathlib.Path(input_file)
    run_dir = pathlib.Path(output_dir) / input_path.stem / algo
    run_dir.mkdir(parents=True, exist_ok=True)
    rr_path = run_dir / f"rr_{input_path.stem}_{algo}.csv"
    features_dir = run_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    return rr_path, features_dir


def run_detection(args: argparse.Namespace) -> pathlib.Path:
    input_path = pathlib.Path(args.file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    first_rows = pd.read_csv(input_path, nrows=100)
    time_column = choose_time_column(first_rows.columns, args.time_column)
    signal_column = choose_signal_column(first_rows, time_column, args.signal_column)
    rr_path, _ = output_paths(str(input_path), args.output_dir, args.algo)

    qrs_detections = []
    for index, chunk in enumerate(pd.read_csv(input_path, chunksize=args.chunk_size)):
        qrs_chunk = detect_chunk_qrs(
            chunk=chunk,
            algo=args.algo,
            sampling_frequency=args.sampling_frequency,
            time_column=time_column,
            signal_column=signal_column,
            max_nan_interpolation=args.max_nan_interpolation,
            min_segment_len=args.min_segment_len,
        )
        qrs_detections.append(qrs_chunk.copy())
        print(f"Chunk {index}: {len(qrs_chunk)} QRS detected")

    rr_df = build_rr_dataframe(qrs_detections, args.sampling_frequency)
    rr_df = rr_df[rr_df["rr_interval"] <= 5000]
    rr_df.to_csv(rr_path, index=False, date_format="%Y-%m-%d_%H:%M:%S.%f%z")
    return rr_path


def main() -> None:
    args = parse_args()

    if args.rr_file:
        rr_path = pathlib.Path(args.rr_file)
        if not rr_path.exists():
            raise FileNotFoundError(f"RR file not found: {rr_path}")
        _, features_dir = output_paths(args.file, args.output_dir, args.algo)
    else:
        rr_path = run_detection(args)
        _, features_dir = output_paths(args.file, args.output_dir, args.algo)

    print(f"RR intervals: {rr_path}")

    if args.skip_features:
        return

    features_path = pathlib.Path(compute_features(str(rr_path), str(features_dir)))
    if features_path.name.startswith("rr_"):
        renamed_path = features_path.with_name("feats_" + features_path.name[len("rr_"):])
        features_path.rename(renamed_path)
        features_path = renamed_path
    print(f"Features: {features_path}")


if __name__ == "__main__":
    main()
