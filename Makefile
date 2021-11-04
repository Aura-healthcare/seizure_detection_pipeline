FOLDER_PATH= .
SRC_PATH=./src
TEST_PATH=./tests

DATA_PATH=data
EXPORT_PATH=./exports

# UTILITIES
clean:
	rm output/db/*csv

test:
	pytest -s -vvv $(TEST_PATH)

coverage:
	pytest --cov=$(SRC_PATH) --cov-report html $(TEST_PATH) 


# PYTHON SCRIPT ON INDIVIDUAL FILES
individual_detect_qrs:
	python3 src/usecase/detect_qrs.py --filepath data/tuh/dev/01_tcp_ar/002/00009578/00009578_s006_t001.edf --method hamilton --exam-id 00009578_s006_t001 --output-folder $(EXPORT_PATH)/individual/res-v0_6

#WIP
example_ecg_qc:
	python3 src/usecase/apply_ecg_qc.py --filepath data/tuh/dev/01_tcp_ar/002/00009578/00009578_s006_t001.edf  --output-folder . --sampling-frequency 1000 --exam-id 00009578_s006_t001

example_hrv:
	python3 src/usecase/compute_hrvanalysis_features.py --rr-intervals-file-path output/rr_intervals/00009578_s006_t001.csv --output-folder .


# All pipeline tasks individually
compute_features:
	. $(FOLDER_PATH)/env/bin/activate; \
	mkdir -p $(EXPORT_PATH); \
	./scripts/10_data_prep/loop_over_TUH_ECG_detector_wrapper.sh  -i $(DATA_PATH) -o $(EXPORT_PATH)/res-v0_6

extract_annotations:
	. $(FOLDER_PATH)/env/bin/activate; \
	mkdir -p $(EXPORT_PATH); \
	./scripts/10_data_prep/loop_over_TUH_Annotation_extractor.sh  -i $(DATA_PATH) -o $(EXPORT_PATH)/annot-v0_6

cardiac_features_computation:
	. $(FOLDER_PATH)/env/bin/activate; \
	./scripts/10_data_prep/loop_over_TUH_Cardiac_features_computation_wrapper.sh -i $(EXPORT_PATH)/res-v0_6 -a $(EXPORT_PATH)/annot-v0_6 -o $(EXPORT_PATH)/feats-v0_6

create_ml_dataset:
	. $(FOLDER_PATH)/env/bin/activate; \
	mkdir -p $(EXPORT_PATH)/ml_dataset; \
	python3 scripts/40_train_model/create_ml_dataset.py -i $(EXPORT_PATH)/feats-v0_6 -o $(EXPORT_PATH)/ml_dataset

train:
	. $(FOLDER_PATH)/env/bin/activate; \
	python3 scripts/40_train_model/train_model.py -i $(EXPORT_PATH)/ml_dataset/df_ml.csv \