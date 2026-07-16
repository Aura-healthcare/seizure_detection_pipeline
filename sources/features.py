from src.usecase.compute_hrvanalysis_features import (
    compute_hrvanalysis_features,
)


def compute_features(rr_file_path: str, output_folder: str) -> str:
    """Compute HRV features from an RR-interval CSV."""
    return compute_hrvanalysis_features(
        rr_intervals_file_path=rr_file_path,
        output_folder=output_folder,
    )

