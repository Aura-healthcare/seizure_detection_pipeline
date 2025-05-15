#!/usr/bin/env sh

source env/bin/activate;
export DATA_PATH=/data/extracts_uncompressed/;
export EXPORT_PATH=/data/rr_interval_studies;

python3 src/usecase/clean_rr_intervals.py  --rr-intervals-file-path /data/1_rr_inteverals/rr_ecg.01-001.csv --output-folder $EXPORT_PATH/2_hrv_features/;
python3 src/usecase/clean_rr_intervals.py  --rr-intervals-file-path /data/1_rr_inteverals/rr_ecg.01-003.csv --output-folder $EXPORT_PATH/2_hrv_features/;
python3 src/usecase/clean_rr_intervals.py  --rr-intervals-file-path /data/1_rr_inteverals/rr_ecg.01-004.csv --output-folder $EXPORT_PATH/2_hrv_features/;
python3 src/usecase/clean_rr_intervals.py  --rr-intervals-file-path /data/1_rr_inteverals/rr_ecg.01-005.csv --output-folder $EXPORT_PATH/2_hrv_features/;
python3 src/usecase/clean_rr_intervals.py  --rr-intervals-file-path /data/1_rr_inteverals/rr_ecg.01-006.csv --output-folder $EXPORT_PATH/2_hrv_features/;
python3 src/usecase/clean_rr_intervals.py  --rr-intervals-file-path /data/1_rr_inteverals/rr_ecg.01-007.csv --output-folder $EXPORT_PATH/2_hrv_features/;
python3 src/usecase/clean_rr_intervals.py  --rr-intervals-file-path /data/1_rr_inteverals/rr_ecg.01-008.csv --output-folder $EXPORT_PATH/2_hrv_features/;
python3 src/usecase/clean_rr_intervals.py  --rr-intervals-file-path /data/1_rr_inteverals/rr_ecg.01-009.csv --output-folder $EXPORT_PATH/2_hrv_features/;
python3 src/usecase/clean_rr_intervals.py  --rr-intervals-file-path /data/1_rr_inteverals/rr_ecg.01-010.csv --output-folder $EXPORT_PATH/2_hrv_features/;
python3 src/usecase/clean_rr_intervals.py  --rr-intervals-file-path /data/1_rr_inteverals/rr_ecg.01-011.csv --output-folder $EXPORT_PATH/2_hrv_features/;
python3 src/usecase/clean_rr_intervals.py  --rr-intervals-file-path /data/1_rr_inteverals/rr_ecg.01-012.csv --output-folder $EXPORT_PATH/2_hrv_features/;
python3 src/usecase/clean_rr_intervals.py  --rr-intervals-file-path /data/1_rr_inteverals/rr_ecg.01-002.csv --output-folder $EXPORT_PATH/2_hrv_features/;
