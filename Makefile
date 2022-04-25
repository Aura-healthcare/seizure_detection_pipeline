FOLDER_PATH= .
SRC_PATH=./src
TEST_PATH=./tests

DATA_PATH=data/PL
EXPORT_PATH=./output

# UTILITIES
# ---------

clean:
	find output -mindepth 1 ! -name README.md -delete

flake8:
	. $(FOLDER_PATH)/env/bin/activate; \
	flake8 --ignore=E402 src/usecase

flake8_all:
	. $(FOLDER_PATH)/env/bin/activate; \
	flake8 --ignore=E402 src/ tests/ dags/

test:
	. $(FOLDER_PATH)/env/bin/activate; \
	pytest -s -vvv $(TEST_PATH)

test_conso:
	. $(FOLDER_PATH)/env/bin/activate; \
		pytest -s -vvv $(TEST_PATH)/test_src_usecase/test_consolidate_feats_and_annot.py


coverage:
	. $(FOLDER_PATH)/env/bin/activate; \
	pytest --cov=$(SRC_PATH) --cov-report html $(TEST_PATH)

# FETCH DATA (FOR AIRFLOW PREPROCESSING)
# -------------
fetch_data:
	. $(FOLDER_PATH)/env/bin/activate; \
	python3 src/usecase/fetch_database.py --data-folder $(DATA_PATH) --export-folder $(EXPORT_PATH)/fetched_data


# PREPROCESSING 
# -------------
# (chose between individual files scripts or all candidate scripts)

# PYTHON SCRIPT ON INDIVIDUAL FILES
individual_detect_qrs:
	. $(FOLDER_PATH)/env/bin/activate; \
	python3 src/usecase/detect_qrs.py --qrs-file-path $(DATA_PATH)/PAT_7/EEG_58_s1.edf --method hamilton --exam-id EEG_58_s1 --output-folder $(EXPORT_PATH)/individual/res-v0_6

individual_compute_hrvanalysis_features:
	. $(FOLDER_PATH)/env/bin/activate; \
	python3 src/usecase/compute_hrvanalysis_features.py --rr-intervals-file-path output/individual/res-v0_6/rr_EEG_58_s1.csv --output-folder $(EXPORT_PATH)/individual/feats-v0_6

individual_consolidate_feats_and_annot:
	. $(FOLDER_PATH)/env/bin/activate; \
	python3 src/usecase/consolidate_feats_and_annot.py --features-file-path output/individual/feats-v0_6/feats_EEG_58_s1.csv --annotations-file-path $(DATA_PATH)/PAT_7_Annotations_EEG_58.tse_bi --output-folder $(EXPORT_PATH)/individual/cons_v0_6


#WIP
example_ecg_qc:
	python3 src/usecase/apply_ecg_qc.py --filepath data/tuh/dev/01_tcp_ar/002/00009578/00009578_s006_t001.edf  --output-folder . --sampling-frequency 1000 --exam-id 00009578_s006_t001


# BASH SCRIPT WRAPPING PYTHON SCRIPTS OVER ALL CANDIDATES
# -------------
bash_detect_qrs:
	. $(FOLDER_PATH)/env/bin/activate; \
	mkdir -p $(EXPORT_PATH); \
	./scripts/bash_pipeline/1_detect_qrs_wrapper.sh  -i $(DATA_PATH) -o $(EXPORT_PATH)/res-v0_6

bash_compute_hrvanalysis_features:
	. $(FOLDER_PATH)/env/bin/activate; \
	./scripts/bash_pipeline/2_compute_hrvanalysis_features_wrapper.sh  -i $(EXPORT_PATH)/res-v0_6 -o $(EXPORT_PATH)/feats-v0_6

bash_consolidate_feats_and_annot:
	. $(FOLDER_PATH)/env/bin/activate; \
	./scripts/bash_pipeline/3_consolidate_feats_and_annot_wrapper.sh  -i $(EXPORT_PATH)/feats-v0_6 -a $(DATA_PATH) -o $(EXPORT_PATH)/cons-v0_6


# CREATION OF THE DATASET
# -----------------------
create_ml_dataset:
	. $(FOLDER_PATH)/env/bin/activate; \
	python3 src/usecase/create_ml_dataset.py --input-folder $(EXPORT_PATH)/cons-v0_6 --output-folder $(EXPORT_PATH)/ml_dataset


# TRAIN OF THE MODEL
# ------------------
train:
	. $(FOLDER_PATH)/env/bin/activate; \
	python3 src/usecase/train_model.py --ml-dataset-path $(EXPORT_PATH)/ml_dataset/df_ml.csv

## VISUALIZATION 
# ------------------
load_ecg:
	python3 visualization/ecg_data_loader.py --pg-host localhost --pg-port 5432 --pg-user postgres --pg-password postgres --pg-database postgres --filepath data/tuh/dev/01_tcp_ar/076/00007633/s003_2013_07_09/00007633_s003_t007.edf

load_rr:
	python3 visualization/rr_intervals_loader.py --pg-host localhost --pg-port 5432 --pg-user postgres --pg-password postgres --pg-database postgres --filepath data/test_data/rr_00007633_s003_t007.csv --exam 00007633_s003_t007

load_annotations:
	python3 visualization/annotations_loader.py --pg-host localhost --pg-port 5432 --pg-user postgres --pg-password postgres --pg-database postgres --annotation-filename data/tuh/dev/01_tcp_ar/076/00007633/s003_2013_07_09/00007633_s003_t007.tse_bi --edf-filename data/tuh/dev/01_tcp_ar/076/00007633/s003_2013_07_09/00007633_s003_t007.edf --exam 00007633_s003_t010
