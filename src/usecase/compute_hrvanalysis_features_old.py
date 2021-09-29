import os

import hrvanalysis as hrv
import pandas as pd
import click

RR_INTERVALS_FOLDER = 'output/clean_rr_intervals'
OUTPUT_FOLDER = 'output/features'


def write_features_csv(features: pd.DataFrame,
                       infos: str) -> str:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    filename = f"{infos}.csv"
    filepath = os.path.join(OUTPUT_FOLDER, filename)
    features.to_csv(filepath, sep=',', index=True)
    return filename


def get_df_hrv_features(rr_intervals: list) -> pd.DataFrame:
    nn_intervals = hrv.get_nn_intervals(rr_intervals)

    # TODO : choose which features
    time_domain_features = hrv.get_time_domain_features(nn_intervals)
    freq_domain_features = hrv.get_frequency_domain_features(nn_intervals)
    geom_features = hrv.get_geometrical_features(nn_intervals)
    csi_cvi_features = hrv.get_csi_cvi_features(nn_intervals)
    poincare_features = hrv.get_poincare_plot_features(nn_intervals)
    entropy_feature = hrv.get_sampen(nn_intervals)

    features_dict = {
        **time_domain_features,
        **freq_domain_features,
        **geom_features,
        **csi_cvi_features,
        **poincare_features,
        **entropy_feature
    }

    return pd.DataFrame([features_dict])


def compute_hrvanalysis_features(rr_intervals_file: str) -> str:
    '''
    Computes features from RR-intervals (from a csv file),
    and writes them in another csv file.
    '''
    df_rr_intervals = pd.read_csv(
        os.path.join(RR_INTERVALS_FOLDER, rr_intervals_file),
        sep=',',
        index_col=0
    )
    rr_intervals = list(df_rr_intervals['rr_interval'])
    df_features = get_df_hrv_features(rr_intervals)
    infos = rr_intervals_file.split('.')[0]
    filename = write_features_csv(df_features, infos)
    return filename


@click.command()
@click.option('--rr-intervals-file', required=True)
def main(rr_intervals_file: str) -> None:
    _ = compute_hrvanalysis_features(rr_intervals_file)


if __name__ == '__main__':
    main()
