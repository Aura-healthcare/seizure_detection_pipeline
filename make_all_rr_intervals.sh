#!/usr/bin/env sh

source env/bin/activate;
export DATA_PATH=/data/extracts_uncompressed/;
export EXPORT_PATH=/data;
# python3 src/usecase/detect_qrs.py --qrs-file-path $DATA_PATH/ecg.01-001.csv --method hamilton  --output-folder $EXPORT_PATH/1_rr_inteverals/;
# python3 src/usecase/detect_qrs.py --qrs-file-path $DATA_PATH/ecg.01-002.csv --method hamilton  --output-folder $EXPORT_PATH/1_rr_inteverals/;
# python3 src/usecase/detect_qrs.py --qrs-file-path $DATA_PATH/ecg.01-003.csv --method hamilton  --output-folder $EXPORT_PATH/1_rr_inteverals/;
# python3 src/usecase/detect_qrs.py --qrs-file-path $DATA_PATH/ecg.01-004.csv --method hamilton  --output-folder $EXPORT_PATH/1_rr_inteverals/;
# python3 src/usecase/detect_qrs.py --qrs-file-path $DATA_PATH/ecg.01-005.csv --method hamilton  --output-folder $EXPORT_PATH/1_rr_inteverals/;
# python3 src/usecase/detect_qrs.py --qrs-file-path $DATA_PATH/ecg.01-006.csv --method hamilton  --output-folder $EXPORT_PATH/1_rr_inteverals/;
# python3 src/usecase/detect_qrs.py --qrs-file-path $DATA_PATH/ecg.01-007.csv --method hamilton  --output-folder $EXPORT_PATH/1_rr_inteverals/;
# python3 src/usecase/detect_qrs.py --qrs-file-path $DATA_PATH/ecg.01-008.csv --method hamilton  --output-folder $EXPORT_PATH/1_rr_inteverals/;
# python3 src/usecase/detect_qrs.py --qrs-file-path $DATA_PATH/ecg.01-009.csv --method hamilton  --output-folder $EXPORT_PATH/1_rr_inteverals/;
# python3 src/usecase/detect_qrs.py --qrs-file-path $DATA_PATH/ecg.01-010.csv --method hamilton  --output-folder $EXPORT_PATH/1_rr_inteverals/;
# python3 src/usecase/detect_qrs.py --qrs-file-path $DATA_PATH/ecg.01-011.csv --method hamilton  --output-folder $EXPORT_PATH/1_rr_inteverals/;
# python3 src/usecase/detect_qrs.py --qrs-file-path $DATA_PATH/ecg.01-012.csv --method hamilton  --output-folder $EXPORT_PATH/1_rr_inteverals/;
#python3 src/usecase/detect_qrs.py --qrs-file-path /data/extracts_uncompressed/ecg.01-013.csv --method hamilton  --output-folder $EXPORT_PATH/1_rr_inteverals/;

# python3 src/usecase/detect_qrs.py --qrs-file-path $DATA_PATH/01-007.egc.sample.csv 01-012.ecg.csv --method hamilton  --output-folder $EXPORT_PATH/1_rr_inteverals/;
# python3 src/usecase/compute_hrvanalysis_features.py --rr-intervals-file-path /data/1_rr_inteverals/rr_ecg.01-001.csv --output-folder $EXPORT_PATH/2_hrv_features/;
# python3 src/usecase/compute_hrvanalysis_features.py --rr-intervals-file-path /data/1_rr_inteverals/rr_ecg.01-002.csv --output-folder $EXPORT_PATH/2_hrv_features/;
# python3 src/usecase/compute_hrvanalysis_features.py --rr-intervals-file-path /data/1_rr_inteverals/rr_ecg.01-003.csv --output-folder $EXPORT_PATH/2_hrv_features/;
# python3 src/usecase/compute_hrvanalysis_features.py --rr-intervals-file-path /data/1_rr_inteverals/rr_ecg.01-004.csv --output-folder $EXPORT_PATH/2_hrv_features/;
# python3 src/usecase/compute_hrvanalysis_features.py --rr-intervals-file-path /data/1_rr_inteverals/rr_ecg.01-005.csv --output-folder $EXPORT_PATH/2_hrv_features/;
python3 src/usecase/compute_hrvanalysis_features.py --rr-intervals-file-path /data/1_rr_inteverals/rr_ecg.01-006.csv --output-folder $EXPORT_PATH/2_hrv_features/;
# python3 src/usecase/compute_hrvanalysis_features.py --rr-intervals-file-path /data/1_rr_inteverals/rr_ecg.01-007.csv --output-folder $EXPORT_PATH/2_hrv_features/;
python3 src/usecase/compute_hrvanalysis_features.py --rr-intervals-file-path /data/1_rr_inteverals/rr_ecg.01-008.csv --output-folder $EXPORT_PATH/2_hrv_features/;
python3 src/usecase/compute_hrvanalysis_features.py --rr-intervals-file-path /data/1_rr_inteverals/rr_ecg.01-009.csv --output-folder $EXPORT_PATH/2_hrv_features/;
# python3 src/usecase/compute_hrvanalysis_features.py --rr-intervals-file-path /data/1_rr_inteverals/rr_ecg.01-010.csv --output-folder $EXPORT_PATH/2_hrv_features/;
# python3 src/usecase/compute_hrvanalysis_features.py --rr-intervals-file-path /data/1_rr_inteverals/rr_ecg.01-011.csv --output-folder $EXPORT_PATH/2_hrv_features/;
# python3 src/usecase/compute_hrvanalysis_features.py --rr-intervals-file-path /data/1_rr_inteverals/rr_01-012.ecg.csv --output-folder $EXPORT_PATH/2_hrv_features/;
python3 src/usecase/compute_hrvanalysis_features.py --rr-intervals-file-path /data/1_rr_inteverals/rr_01-013.ecg.csv --output-folder $EXPORT_PATH/2_hrv_features/;
